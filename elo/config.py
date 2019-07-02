#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: config.py
Author: zlamberty
Created: 2017-12-02

Description:
    configuration for certain implementations of ELO scoring (e.g. 538's NFL
    football ELO rankings)

Usage:
    <usage>

"""

import logging


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

LOGGER = logging.getLogger(__name__)


# ----------------------------- #
#   configurations              #
# ----------------------------- #

NFL_538 = {
    'k': 20,
    'ptscale': 400,
    'hfa': 65,
    'base': 10,
    'a0': 2.2,
    'a1': 0.001,
    'a2': 2.2,
}

# we obtained values of k, hfa from here:
#  https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
NBA_538 = {
    'k': 20,
    'ptscale': 400,
    'hfa': 100,
    'base': 10,
    'a0': 3,
    'a1': 0.006,
    'a2': 7.5,
    'power': 0.8
}
