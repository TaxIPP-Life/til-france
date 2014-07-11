#! /usr/bin/env python
# -*- coding: utf-8 -*-


# TaxIPP-Life BaseModel (TIL-BaseModel) -- A microsimulation model over the life-cycle
# By: TaxIPP-Life (TIL) Team <alexis.eidelman.pro@gmail.com>
#
# Copyright (C) 2011, 2012, 2013, 2014 TaxIPP-Life (TIL) Team
# (https://github.com/TaxIPP-Life/Til-BaseModel)
#
# This file is part of TaxIPP-Life (TIL).
#
# TaxIPP-Life (TIL) is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# TaxIPP-Life (TIL) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""TaxIPP-Life (TIL) is microsimulation model over the life-cycle

TaxIPP-Life (TIL) is based on OpenFisca (www.openfisca.fr) and liam2 (http://liam2.plan.be)
"""


from setuptools import setup, find_packages


classifiers = """\
Development Status :: 2 - Pre-Alpha
License :: OSI Approved :: GNU Affero General Public License v3
Operating System :: POSIX
Programming Language :: Python
Topic :: Scientific/Engineering :: Information Analysis
"""

doc_lines = __doc__.split('\n')


setup(
    name = 'Til-BaseModel',
    version = '0.1dev',

    author = 'TaxIPP-Life (TIL) Team',
    author_email = 'alexis.eidelman.pro@gmail.com',
    classifiers = [classifier for classifier in classifiers.split('\n') if classifier],
    description = doc_lines[0],
    keywords = 'benefit microsimulation social tax life-cycle',
    license = 'http://www.fsf.org/licensing/licenses/agpl-3.0.html',
    long_description = '\n'.join(doc_lines[2:]),
    url = 'https://github.com/TaxIPP-Life/Til-BaseModel',
    install_requires = [
        'numpy',
        ],
    message_extractors = {
        'til_base_model': [
            ('**.py', 'python', None),
            ],
        },
    packages = find_packages(),
    zip_safe = False,
    )
