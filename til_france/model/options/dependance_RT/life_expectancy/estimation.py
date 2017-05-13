# -*- coding: utf-8 -*-

import os
import pandas as pd
import pkg_resources


assets_path = config_files_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'assets',
    )

matrix_path = os.path.join(assets_path, '3c', 'Proba_transition_3C.xlsx')

df = pd.read_excel(matrix_path, index_col = 0, parse_cols = 'A:G').reset_index()

matrix_3c = df.loc[1:6]
matrix_3c.columns = ['initial'] + range(0, 6)
matrix_3c = (matrix_3c
    .set_index('initial')
    .fillna(0)
    )
paquid = df.loc[11:16]
paquid.columns = ['initial'] + range(0, 6)
paquid = (paquid
    .set_index('initial')
    .fillna(0)
    )


