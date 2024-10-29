#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Rhizome
# Version beta 0.0, August 2023
# Property of IBM Research, Accelerated Discovery
#

"""
PLEASE NOTE THIS IMPLEMENTATION INCLUDES THE ORIGINAL SOURCE CODE (AND SOME ADAPTATIONS)
OF THE MHG IMPLEMENTATION OF HIROSHI KAJINO AT IBM TRL ALREADY PUBLICLY AVAILABLE. 
THIS MIGHT INFLUENCE THE DECISION OF THE FINAL LICENSE SO CAREFUL CHECK NEEDS BE DONE. 
"""

""" Title """

__author__ = "Hiroshi Kajino <KAJINO@jp.ibm.com>"
__copyright__ = "(c) Copyright IBM Corp. 2017"
__version__ = "0.1"
__date__ = "Dec 11 2017"

from abc import ABCMeta, abstractmethod

class GraphGrammarBase(metaclass=ABCMeta):
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def sample(self):
        pass
