#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: elo
Author: zlamberty
Created: 2016-11-30

Description:
    some details at the following links:
    1. https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
    2. https://doubleclix.wordpress.com/2015/01/20/the-art-of-nfl-ranking-the-elo-algorithm-and-fivethirtyeight/
    3. https://fivethirtyeight.com/features/nfl-elo-ratings-are-back/

    and most importantly,
    https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py

Usage:
    <usage>

"""

import logging

import pandas as pd
import scipy as sp
import tqdm

from . import config


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

LOGGER = logging.getLogger(__name__)


# ----------------------------- #
#   Main routine                #
# ----------------------------- #

class Elo(object):
    """a utility model implementing the Elo rating system"""
    def __init__(self, k, ptscale, hfa=0, base=10):
        """instantiate a new Elo rating model

        args:
            k (float): the maximum number of points a player can gain as the
                result of a single victory
            ptscale (float): this is the rough point scale defining differences
                between teams (basically, this defines the decay rate of Elo
                rating values in terms of expected results in head-to-head
                matches)
            hfa (float): the value assigned to home-field advantage (default: 0)
            base (float): the base exponent in the expected score calculation
                (default: 10)

        returns:
            an instance of the Elo class

        raises:
            None

        """
        self.k = k
        self.ptscale = ptscale
        self.hfa = hfa
        self.base = base
        self.has_homefield = self.hfa != 0

    # matchup functions
    def elo_diff(self, r0, r1, han='H'):
        """the difference between Elo rankings

        this is straight forward except for the possible addition of a home
        field advantage delta, so we incorporate that here.

        args:
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: possibly hfa-adjusted difference between Elo

        raises:
            None

        """
        hansign = sp.where(
            han == 'H',
            1,               # home
            sp.where(
                han == 'A',
                -1,          # away
                0            # neutral
            )
        )
        return (r0 - r1) + hansign * self.hfa

    def expected_score(self, r0, r1, han='H'):
        """the expected score (0 to 1) of `r0` in a match against `r1`

        the score is effectively a probability of victory, and ranges from 0 to
        1. in the real world, a loss is a 0, a win is a 1, and a tie is a 0.5.
        we allow for homefield advantage via the instance attribute `hfa`
        (measured in Elo pts) and a boolean indicating whether or not r0 is the
        home team

        args:
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the expected score (between 0 and 1) of `r0` winning this
                matchup

        """
        ed = self.elo_diff(r0, r1, han)
        return 1 / (1 + self.base ** (-1 * ed / self.ptscale))

    def update_score(self, r0, r1, realizedscore, deltapts, han='H'):
        """update a current Elo ranking based on the results of a match.

        `deltapts`` is a directional quantity, so if team 0 lost to team 1,
        `deltapts` needs to represent this by being negative.

        this function is vectorized, so for each of the arguments below, we can
        accept a pandas.Series object containing comparable dtypes.

        args:
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            realizedscore (float): the real-life point value of the match (0 for
                a loss, 1 for a win, and 0.5 for a tie)
            deltapts (float): the difference in real-world points in the game
                between teams 0 and 1 (with Elo rankings `r0` and `r1`, resp).
                this must be directional (positive for a win, negative for a
                loss)
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the updated Elo ranking

        raises:
            None

        """
        expectedscore = self.expected_score(r0, r1, han)
        return r0 + self.k * mov * (realizedscore - expectedscore)

    def elo_history(self, scoredf, idcol='id', opp_idcol=None, scorecol='score',
                    opp_scorecol=None, seasoncol='season',
                    matchupperiodcol='matchup', hancol='han',
                    reversioncoef=0.75, elomean=1500, innotebook=False):
        """calculate elo scores from a dataframe of historical game performances

        in order to implement this, we have requirements on the provided score
        dataframe `scoredf`. First, each record is one team's perspective for a
        match (that is, each match results in two records, one for each
        opponent). each match record must have the following fields (names are
        parameterized but the meaning must be consistent):

            id: a column indicating a unique id for a team (this cannot
                change over time, but can blink in or out of existence (e.g. a
                team is no longer in a league, or an expansion team))
            opp_id: the id of the opponent in this matchup.
            score: the competition-specific score obtained by the primary
                team (e.g. in the nfl 14 pts, in the nba 102 pts)
            opp_score: the competition-specific score obtained by the
                opponent
            season: some ordered indicator of the season (does not have to
                have multiple values, but does have to exist)
            matchupperiod: some ordered indicator of the matchup period
                within a given season (e.g. week number in the nfl, day within
                the nba)
            ishome: categorical feature indicating whether or not the main team
                for this record was home, away, or neutral ('H', 'A', 'N')
            reversioncoef: the inter-season reversion coefficient
            elomean: the expected long-term mean value of elo scores

        args:
            scoredf (pandas.DataFrame): a dataframe of season scores formatted
                as described above
            idcol (str): the name of the column indicating the team's id
                (default: 'id')
            opp_idcol (str): the name of the column indicating the opponent's
                id. if a value of `None` is provided, we will assume it is
                `opp_{idcol:}` (default: None)
            scorecol (str): the name of the column indicating the
                competition-specific score obtained by the primary team
                (default: 'score')
            opp_scorecol (str): the name of the column indicating the
                competition-specific score obtained by the opponent. if a value
                of `None` is provided, we will assume it is `opp_{scorecol:}`
                (default: None)
            seasoncol (str): the name of the column indicating the season of the
                matchup (must be monotonic) (default: 'season')
            matchupperiodcol (str): the name of the column indicating the
                matchup period within a given season (must be monotonic within a
                given season) (default: 'matchup')
            hancol (str): the name of the column containing the categorical feature
                indicating whether or not the main team for this record was
                home, away, or neutral (default: 'han')
            reversioncoef (float): the inter-season reversion coefficient (must
                be between 0 and 1). (default: 0.75)
            elomean (float): the expected long-term mean value of elo scores
                (default: 1500)
            innotebook (bool): completely cosmetic for tqdm purposes, indicates
                whether or not or progress bar should be text (console) or
                javascript (notebook) based. since text works for both
                environments: (default: False)

        returns:
            pandas.DataFrame: a dataframe with elo history keyed by season,
                matchup, and team id

        raises:
            AssertionError

        """
        # reversion must be well defined
        assert 0 <= reversioncoef <= 1

        # set a tqdm operator
        mytqdm = tqdm.tqdm_notebook if innotebook else tqdm.tqdm

        # fix the opponent columns
        opp_idcol = opp_idcol or 'opp_{}'.format(idcol)
        opp_scorecol = opp_scorecol or 'opp_{}'.format(scorecol)

        # sorting makes our groupby life easier
        scoredf = scoredf.sort_values(by=[seasoncol, matchupperiodcol, idcol])

        # we will also eventually need the win/lose/tie value (elo values, so
        # 1/0/0.5).
        scoredf.loc[:, 'score_delta'] = scoredf[scorecol] - scoredf[opp_scorecol]
        scoredf.loc[:, 'wlt'] = sp.where(
            scoredf[scorecol] > scoredf[opp_scorecol],
            1,
            sp.where(scoredf[scorecol] < scoredf[opp_scorecol], 0, 0.5)
        )

        dfelo = pd.DataFrame()
        for (season, scoredfnow) in mytqdm(scoredf.groupby(seasoncol)):
            # elo init or revert to mean
            dfelonow = pd.DataFrame({idcol: scoredfnow[idcol].unique()})
            dfelonow.loc[:, seasoncol] = season
            dfelonow.loc[:, matchupperiodcol] = scoredfnow[matchupperiodcol].min()

            if dfelo.empty:
                # init an elo dataframe
                dfelonow.loc[:, 'elo'] = elomean
            else:
                # mean reversion! first, we need the elo at the end of the prev
                # season
                eloprev = dfelo[dfelo[seasoncol] < season]
                eloprev = eloprev[eloprev[seasoncol] == eloprev[seasoncol].max()]
                eloprev = eloprev[
                    eloprev[matchupperiodcol] == eloprev[matchupperiodcol].max()
                ]

                # now, regress that to the mean by our reversion factor and join
                # that in for each idcol value. this is a complicated way of
                # writing
                eloprev.loc[:, 'elo'] = (
                    elomean + reversioncoef * (eloprev.elo - elomean)
                )

                dfelonow = dfelonow.merge(
                    eloprev[[idcol, 'elo']],
                    how='left',
                    left_on=[idcol],
                    right_on=[idcol]
                ).fillna(elomean)

            dfelo = dfelo.append(dfelonow).reset_index(drop=True)

            # iterate through weeks *in this year*. I would love to do groupby but
            # i need to know what the next matchup is every time and in case there's
            # a skip in the matchups...
            matchups = scoredfnow[matchupperiodcol].unique()
            for (i, matchup) in enumerate(mytqdm(matchups, leave=False)):
                # pull up the scores and elo rankings for this (season,
                # matchup)
                dfweek = scoredfnow[
                    scoredfnow[matchupperiodcol] == matchup
                ].copy()
                assert not dfweek.empty

                thiselo = dfelo[
                        (dfelo[seasoncol] == season)
                    & (dfelo[matchupperiodcol] == matchup)
                ]

                # join the current elo values for both teams with the scores
                dfwelo = dfweek.merge(
                    thiselo,
                    how='right',
                    on=[idcol, seasoncol, matchupperiodcol]
                )
                dfwelo = dfwelo.merge(
                    thiselo,
                    how='left',
                    left_on=[opp_idcol, seasoncol, matchupperiodcol],
                    right_on=[idcol, seasoncol, matchupperiodcol],
                    suffixes=('', '_opp')
                )

                # iterate the matchup number (we're about to calculate
                # *next week's* elo ranking)
                try:
                    dfwelo.loc[:, matchupperiodcol] = matchups[i + 1]
                except IndexError:
                    dfwelo.loc[:, matchupperiodcol] = matchup + 1

                # magic time
                dfwelo.loc[:, 'elo'] = self.update_score(
                    r0=dfwelo.elo,
                    r1=dfwelo.elo_opp,
                    realizedscore=dfwelo.wlt,
                    deltapts=dfwelo.score_delta,
                    han=dfwelo[hancol].fillna('N')
                ).fillna(dfwelo.elo)

                dfelo = dfelo.append(
                    dfwelo[[idcol, matchupperiodcol, seasoncol, 'elo']]
                )

        return dfelo

    # other utilities
    def q_score(self, r):
        """the Q score (logarithmic value) of a team with Elo ranking `r`

        args:
            r (float): Elo ranking

        returns:
            float: the logarithmic raw score

        raises:
            None

        """
        return self.base ** (r / self.ptscale)

    def point_spread(self, r0, r1, spreadscale, han='H'):
        """given a linear factor relating elo differentials to points spreads,
        do the simple math

        args:
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            spreadscale (floag): the linear relationship between point spreads
                and Elo differentials (empirical quantity)
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: estimated point spread

        raises:
            None

        """
        return self.elo_diff(r0, r1, han) / spreadscale


class EloWMarginOfVictory(Elo):
    """a utility model implementing the Elo rating system

    examples:
        to create the 538 NFL elo model

        >>> nflelo = EloWMarginOfVictory(**elo.config.NFL_538)

    """
    def __init__(self, k, ptscale, a0, a1, a2, hfa=0, base=10, power=0.8,
                 movtype='nfl'):
        """instantiate a new Elo rating model with margin of victory multiplier

        args:
            k (float): the maximum number of points a player can gain as the
                result of a single victory
            ptscale (float): this is the rough point scale defining differences
                between teams (basically, this defines the decay rate of Elo
                rating values in terms of expected results in head-to-head
                matches)
            a0 (float): empirically derived constants for scaling the mov
                multiplier
            a1 (float): empirically derived constants for scaling the mov
                multiplier
            a2 (float): empirically derived constants for scaling the mov
                multiplier
            hfa (float): the value assigned to home-field advantage (default: 0)
            base (float): the base exponent in the expected score calculation
                (default: 10)
            power (float): the power in a fitted margin of victory multiplier
                calculation (only used for the nba movtype)
            movtype (str): the type of margin of victory multiplier utilized by
                fivethirtyeight is *different* between the nfl and nba models,
                and I'm not sure why. currently supported methods are `nfl` and
                `nba`

        returns:
            an instance of the EloWMarginOfVictory class

        raises:
            None

        """
        super().__init__(k, ptscale, hfa, base)
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.power = power
        self.movtype = movtype

    def update_score(self, r0, r1, realizedscore, deltapts, han='H'):
        """update a current Elo ranking based on the results of a match.

        same as the definition for the non-mov version except for one additional
        scaling factor (mov)

        note: `deltapts`` is a directional quantity, so if team 0 lost to team 1,
        `deltapts` needs to represent this by being negative

        args:
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            realizedscore (float): the real-life point value of the match (0 for
                a loss, 1 for a win, and 0.5 for a tie)
            deltapts (float): the difference in real-world points in the game
                between teams 0 and 1 (with Elo rankings `r0` and `r1`, resp)
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the updated Elo ranking

        raises:
            None

        """
        expectedscore = self.expected_score(r0, r1, han)
        mov = self.margin_of_victory_multiplier(deltapts, r0, r1, han)
        return r0 + self.k * mov * (realizedscore - expectedscore)

    def margin_of_victory_multiplier(self, deltapts, r0, r1, han='H'):
        """a multiplier to amplify how badly a team beat the other team.

        the actual implementation here depends on the attribute `movtype`

        note: this is a directional quantity, so if team 0 lost to team 1,
        `deltapts` needs to represent this by being negative.

        args:
            deltapts (float): the difference in real-world points in the game
                between teams 0 and 1 (with Elo rankings `r0` and `r1`, resp)
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the margin of victory multiplier

        raises:
            ValueError

        """
        if self.movtype == 'nfl':
            return self._mov_nfl(deltapts, r0, r1, han)
        elif self.movtype == 'nba':
            return self._mov_nba(deltapts, r0, r1, han)
        else:
            raise ValueError("movtype must be one of 'nfl' or 'nba'")

    def _mov_nfl(self, deltapts, r0, r1, han='H'):
        """margin of victory multiplier used for the nfl elo ranking.

        see the motivating definition here:
        https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py

        args:
            deltapts (float): the difference in real-world points in the game
                between teams 0 and 1 (with Elo rankings `r0` and `r1`, resp)
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the margin of victory multiplier

        raises:
            None

        """
        return (
            sp.log(sp.maximum(deltapts, 1) + 1)
            * (self.a0 / (self.elo_diff(r0, r1, han) * self.a1 + self.a2))
        )

    def _mov_nba(self, deltapts, r0, r1, han='H'):
        """margin of victory multiplier used for the nfl elo ranking

        see the motivating definition here:
        https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
        (it's hidden in footnote 2 as of 2018-03)

        args:
            deltapts (float): the difference in real-world points in the game
                between teams 0 and 1 (with Elo rankings `r0` and `r1`, resp)
            r0 (float): the Elo ranking of team 0
            r1 (float): the Elo ranking of team 1
            han (str): one of 'H', 'A', or 'N', indicating whether or not
                the team with Elo ranking `r0` is home, away, or neutral
                (default: 'H')

        returns:
            float: the margin of victory multiplier

        raises:
            None

        """
        try:
            dpabs = deltapts.abs()
        except AttributeError:
            dpabs = abs(deltapts)
        return (
            (dpabs + self.a0) ** self.power
            / (self.elo_diff(r0, r1, han) * self.a1 + self.a2)
        )
