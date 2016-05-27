# -*- coding: utf-8 -*-

import logging
import numpy as np

#
# /*************************************************/
# /* Chiffrement des axes AGGIR sur le             */
# /* fichier d'enquête en domicile ordinaire       */
# /*************************************************/
# /*  gprojet\gdtincap\aggir\girv3.sas             */
# /*************************************************/
# /* Remarque : pour un travail sur le fichier en  */
# /* institution, il faudra adapter ce programme   */
# /* au questionnement qui est différent           */
# /*********************************************** */

# data aggir; merge hid99.modb_c aggir13; by ident numind;


from openfisca_survey_manager.survey_collections import SurveyCollection


log = logging.getLogger(__name__)


def get_dataframe():

    survey_collection = SurveyCollection.load(collection = 'hid')
    survey = survey_collection.get_survey('hid_domicile_1999')
    modb = survey.get_values(table = 'modb_c')
    modb = build_axes(modb)

    result = modb[['ident', 'numind'] + ['axe{}'.format(i) for i in range(1, 11)]].copy()
    result['rang'] = 0
    result['gir'] = 0

    for group, items in groups.iteritems():
        print group
        result['valeur'] = 0
        for i in range(1, 11):
            for axe in items['axes']:
                name = axe['name']
                values = axe['values']
                # print i, name, values

                result.loc[
                    result['axe{}'.format(i)] == name,
                    'valeur',
                    ] += values[i]
        print 'max:', result.valeur.max()

        for seuil in items['seuils']:
            gir = seuil.get('gir')
            if gir is None:
                continue
            rang = seuil['rang']
            seuil_valeur = seuil['valeur']
            result.loc[
                (result.valeur >= seuil_valeur) & (result.rang == 0),
                'rang'] = rang
            result.loc[
                (result.valeur >= seuil_valeur) & (result.gir == 0),
                'gir'] = gir
            print rang, gir, seuil_valeur
            print result.gir.value_counts().sort_index()
            # print result.rang.value_counts().sort_index()

    mindiv = survey.get_values(table = 'mindiv_c', variables = ['ident', 'numind', 'poidscor'])
    result.gir.value_counts(dropna = False)
    result = result.merge(mindiv, on = ['ident', 'numind'])
    print result.groupby(['gir'])['poidscor'].sum() / 1e3
    return result


def build_axes(modb):
    modb.btoi1.value_counts(dropna = False)
    modb.bale1.value_counts(dropna = False)

    # if btoi1 ne ' ' and bale1 ne ' ';
    #        axe1='9';
    #    if bcoh1 in ('0','4') or bcoh2='3' then axe1='C';
    #    else if bcoh2='2' then axe1='B';
    #    else axe1='A';

    modb.loc[(modb.btoi1 == 0) & (modb.bale1 == 0), 'axe1'] = None
    modb.loc[
        modb.btoi1 != 0,
        'axe1'
        ] = 'A'
    modb.loc[
        (modb.bcoh2 == 2),
        'axe1',
        ] = 'B'
    modb.loc[
        modb.bcoh1.isin([0, 4]) | (modb.bcoh2 == 3),
        'axe1',
        ] = 'C'

    print modb.axe1.value_counts(dropna = False)
    assert modb.axe1.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()

     # /* 2. Axe 2 : Orientation */
     # /**************************/
     #       axe2='9';
     #     if bori1='3' or bori2='4' then axe2='C';
     #     else if bori2='3' or bori1='2' then axe2='B';
     #     else axe2='A';
     #      run;
     # *** 16620 obs : le GIR n'est pas calculable pour les enfants de moins de 5 ans ;
    # J'en trouve 16610

    modb['axe2'] = 'A'
    modb.loc[
        (modb.bori2 == 3) | (modb.bori1 ==2),
        'axe2',
        ] = 'B'
    modb.loc[
        (modb.bori1 == 3) | (modb.bori2 == 4),
        'axe2',
        ] = 'C'

    modb.loc[modb.btoi1 == 0, 'axe2'] = None
    print modb.axe2.value_counts(dropna = False)
    assert modb.axe2.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()


     # /* I. Passage des réponses au questionnaire */
     # /*     aux "notes" des 10 axes AGGIR        */
     # /********************************************/
     # data aggir; merge aggir (keep=ident numind axe1-axe2 in=x)
     #                   hid99.modb_c; by ident numind;
     #    if x=0 then enfant=1; else enfant=2;

     # /* 3. Axe 3 : Toilette */
     # /***********************/
     #       axe3='9';
     #       if btoi1 in ('0','4','6') then axe3='C';
     #       else if btoi1='5' then axe3='B';             * ou mettre le 3;
     #       else if btoi1 in ('1','2','3') then axe3='A';


    modb['axe3'] = None
    modb.loc[
        modb.btoi1.isin([1, 2, 3]),
        'axe3',
        ] = 'A'
    modb.loc[
        modb.btoi1 == 5,
        'axe3',
        ] = 'B'
    modb.loc[
        modb.btoi1.isin([0, 4, 6]),
        'axe3',
        ] = 'C'

    print modb.axe3.value_counts(dropna = False)
    assert modb.axe3.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()

     # /* 4. Axe 4 : Habillage */
     # /************************/
     #
     #       axe4='9';
     #       if bhab1 in ('0','6') then axe4='C';
     #       else if bhab1 in ('4','5') then axe4='B';
     #       else axe4='A';         * ou mettre le 3;


    modb['axe4'] = 'A'
    modb.loc[
        modb.bhab1.isin([4, 5]),
        'axe4',
        ] = 'B'
    modb.loc[
        modb.bhab1.isin([0, 6]),
        'axe4',
        ] = 'C'

    print modb.axe4.value_counts(dropna = False)
    assert modb.axe4.value_counts(dropna = False).index.isin(['A', 'B', 'C']).all()


     # /* 5. Axe 5 : Alimentation */
     # /***************************/
     #       axe5='9';
     #       if bali1='0' then axe5='C';
     #       else do;
     #         if bali1 in ('4','5') and bali2 in ('0','4') then serv='C';
     #         else if bali1 in ('4','5') or bali2 in ('0','4') then serv='B';        * ou mettre le 3;
     #         else if bali1 in ('1','2','3') and bali2 in ('1','2','3') then serv='A';
     #         if bali3='4' then mang='C';
     #         else if bali3='3' then mang='B';
     #         else if bali3 in ('1','2') then mang='A';
     #         if serv='C' and mang='C' then axe5='C';
     #         else if serv='C' and mang='B' then axe5='C';
     #         else if serv='B' and mang='C' then axe5='C';
     #         else if serv='A' and mang='A' then axe5='A';
     #         else axe5='B';
     #       end;

    modb.bali1.value_counts(dropna = False)
    modb.bali2.value_counts(dropna = False)
    modb.bali3.value_counts(dropna = False)

    # serv
    modb['serv'] = None

    modb.loc[
        modb.bali1.isin([1, 2, 3]) & modb.bali2.isin([1, 2, 3]),
        'serv',
        ] = 'A'
    modb.loc[
        modb.bali1.isin([4, 5]) | modb.bali2.isin([0, 4]),
        'serv',
        ] = 'B'
    modb.loc[
        modb.bali1.isin([4, 5]) & modb.bali2.isin([0, 4]),
        'serv',
        ] = 'C'

    # mang
    modb['mang'] = None
    modb.loc[modb.bali3.isin([1, 2]), 'mang'] = 'A'
    modb.loc[modb.bali3 == 3, 'mang'] = 'B'
    modb.loc[modb.bali3 == 4, 'mang'] = 'C'

    # sum up
    modb.loc[
        (modb.serv == 'A') & (modb.mang == 'A'),
        'axe5',
        ] = 'A'
    modb.loc[
        (modb.serv == 'B') & (modb.mang == 'B'),
        'axe5',
        ] = 'B'
    modb.loc[
        (modb.serv == 'C') & (modb.mang == 'C'),
        'axe5',
        ] = 'C'
    modb.loc[
        (modb.serv == 'C') & (modb.mang == 'B'),
        'axe5',
        ] = 'C'
    modb.loc[
        modb.bali1 == 0,
        'axe5',
        ] = 'C'
    modb.loc[
        modb.axe5.isnull(),
        'axe5',
        ] = 'B'

    print modb.axe5.value_counts(dropna = False)
    assert modb.axe5.value_counts(dropna = False).index.isin(['A', 'B', 'C']).all()


     # /* 6. Axe 6 : Elimination */
     # /**************************/
     #       axe6='9';
     #       if beli1='0' or beli3='5' then axe6='C';
     #       else if beli1='5' then do;
     #            if beli3=' ' then axe6='B';
     #            else axe6='C';
     #       end;
     #       else if beli1='4' or beli3='4' then axe6='B';             * ou mettre les 3 ;
     #       else if beli1 in ('1','2','3','9')
     #            and beli3 in ('1','2','3','9') then axe6='A';


    modb.beli1.value_counts(dropna = False)
    modb.beli3.value_counts(dropna = False)

    modb['axe6'] = None
    modb.loc[
        modb.beli1.isin([1, 2, 3, 9]) & modb.beli3.isin([1, 2, 3, 9]),
        'axe6'
        ] = 'A'
    modb.loc[
        (modb.beli1 == 5) & (modb.beli3 == 0),
        'axe6'
        ] = 'B'
    modb.loc[
        (modb.beli1 == 4) | (modb.beli3 == 4),
        'axe6'] = 'B'
    modb.loc[
        (modb.beli1 == 0) | (modb.beli3 == 5),
        'axe6'
        ] = 'C'
    modb.loc[
        (modb.beli1 == 5) & (modb.beli3 != 0),
        'axe6'
        ] = 'C'

    print modb.axe6.value_counts(dropna = False)
    assert modb.axe6.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()


     # /* 7. Axe 7 : Transferts */
     # /*************************/
     #       axe7='9';
     #       if bmob1='1' or (btra1='4' and btra2='4') or btra2='0' then axe7='C';
     #       else if btra1='4' or (btra2='4') or (btra1='3' and btra2='3') then axe7='B';
     #       else if btra1 in ('1','2','3','9')
     #            and btra2 in ('1','2','3') then axe7='A';


    modb.bmob1.value_counts(dropna = False)
    modb.btra1.value_counts(dropna = False)
    modb.btra2.value_counts(dropna = False)

    modb['axe7'] = None
    modb.loc[
        modb.btra1.isin([1, 2, 3, 9]) & modb.btra2.isin([1, 2, 3]),
        'axe7',
        ] = 'A'
    modb.loc[
        (modb.btra1 == 4) | (modb.btra2 == 4) | ((modb.btra1 == 3) & (modb.btra2 == 3)),
         'axe7',
         ] = 'B'
    modb.loc[
        (modb.bmob1 == 1) | ((modb.btra1 ==  4) & (modb.btra2 == 4)) | modb.btra2 == 0,
         'axe7',
         ] = 'C'

    print modb.axe7.value_counts(dropna = False)
    assert modb.axe7.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()


     # /* 8. Axe 8 : Déplacements à l'intérieur */
     # /*****************************************/
     #       axe8='9';
     #       if bmob1 in ('1','2') or bdpi1='3' then axe8='C';
     #       else if bdpi1='2'
     #               or bdpi2 in ('4','5') then axe8='B';
     #       else if bdpi1='1' then axe8='A';


    modb.bmob1.value_counts(dropna = False)
    modb.bdpi1.value_counts(dropna = False)
    modb.bdpi2.value_counts(dropna = False)

    modb['axe8'] = None
    modb.loc[modb.bdpi1 == 1, 'axe8'] = 'A'
    modb.loc[
         (modb.bdpi1 == 2) | modb.bdpi2.isin([4, 5]),
        'axe8',
        ] = 'B'
    modb.loc[
         modb.bmob1.isin([1, 2]) | (modb.bdpi1 ==  3),
        'axe8',
        ] = 'C'

    print modb.axe8.value_counts(dropna = False)
    assert modb.axe8.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()


     # /* 9. Axe 9 : Déplacements à l'extérieur */
     # /*****************************************/
     #       axe9='9';
     #       if bdpe2=0 or bmob1 in ('1','2','3') then axe9='C';
     #       else if 0<bdpe2<=100 then axe9='B';
     #       else if 100<bdpe2 then axe9='A';

    modb.bdpe2.value_counts(dropna = False)
    modb.bmob1.value_counts(dropna = False)

    modb['axe9'] = None
    modb.loc[
         modb.bdpe2 > 100,
        'axe9',
        ] = 'A'
    modb.loc[
         (0 < modb.bdpe2) & (modb.bdpe2 <= 100),
        'axe9',
        ] = 'B'
    modb.loc[
         (modb.bdpe2 == 0) | modb.bmob1.isin([1, 2, 3]),
        'axe9',
        ] = 'C'

    print modb.axe9.value_counts(dropna = False)
    assert modb.axe9.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()


     # /* 10. Axe 10 : Communication à distance */
     # /*****************************************/
     #       axe10='9';
     #       if bale1 in ('2','3','4') then axe10='C';
     #       else if bale1='1' then axe10='B';
     #       else if bale1 in ('0','9') then axe10='A';
     # run;

    modb.bale1.value_counts(dropna = False)

    modb['axe10'] = None
    modb.loc[
         modb.bale1.isin([0, 9]),
        'axe10',
        ] = 'A'
    modb.loc[
         modb.bale1 == 1,
        'axe10',
        ] = 'B'
    modb.loc[
        modb.bale1.isin([2, 3, 4]),
        'axe10',
        ] = 'C'

    print modb.axe10.value_counts(dropna = False)
    assert modb.axe10.value_counts(dropna = False).index.isin(['A', 'B', 'C', np.nan]).all()
    return modb


#    /*******************************************************/
#    /* Programme de calcul des groupes iso-ressources      */
#    /* conforme à l'algorithme publià au J.O. du 30/04/97  */
#    /*******************************************************/
#    /*   gprojet\gdtincap\prog\GIRCALC2.SAS                */
#    /********************** (08/02/99) *********************/
#    /* Préparation du fichier standard */
#    /***********************************/
#    data agirgrp;
#           set aggir (keep=ident numind axe1-axe10 enfant) ; by ident numind;
#        if axe1 ne '9';
#        array axei(10) axe1-axe10;
#        /* Groupe A */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+2000;
#            else if i=2 then valeur=valeur+1200;
#            else if i=3 then valeur=valeur+40;
#            else if i=4 then valeur=valeur+40;
#            else if i=5 then valeur=valeur+60;
#            else if i=6 then valeur=valeur+100;
#            else if i=7 then valeur=valeur+800;
#            else if i=8 then valeur=valeur+200;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+16;
#            else if i=4 then valeur=valeur+16;
#            else if i=5 then valeur=valeur+20;
#            else if i=6 then valeur=valeur+16;
#            else if i=7 then valeur=valeur+120;
#            else if i=8 then valeur=valeur+32;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=4380 then do; rang=1; gir=1; end;
#          else if valeur>=4140 then do; rang=2; gir=2; end;
#          else if valeur>=3390 then do; rang=3; gir=2; end;
#        if valeur>=3390 then go to FIN;
#        /* Groupe B */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+1500;
#            else if i=2 then valeur=valeur+1200;
#            else if i=3 then valeur=valeur+40;
#            else if i=4 then valeur=valeur+40;
#            else if i=5 then valeur=valeur+60;
#            else if i=6 then valeur=valeur+100;
#            else if i=7 then valeur=valeur+800;
#            else if i=8 then valeur=valeur-80;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+320;
#            else if i=2 then valeur=valeur+120;
#            else if i=3 then valeur=valeur+16;
#            else if i=4 then valeur=valeur+16;
#            else if i=5 then valeur=valeur+0;
#            else if i=6 then valeur=valeur+16;
#            else if i=7 then valeur=valeur+120;
#            else if i=8 then valeur=valeur-40;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=2016 then do; rang=4; gir=2; end;
#        if valeur>=2016 then go to FIN;
#        /* gir C */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+40;
#            else if i=4 then valeur=valeur+40;
#            else if i=5 then valeur=valeur+60;
#            else if i=6 then valeur=valeur+160;
#            else if i=7 then valeur=valeur+1000;
#            else if i=8 then valeur=valeur+400;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+16;
#            else if i=4 then valeur=valeur+16;
#            else if i=5 then valeur=valeur+20;
#            else if i=6 then valeur=valeur+20;
#            else if i=7 then valeur=valeur+200;
#            else if i=8 then valeur=valeur+40;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=1700 then do; rang=5; gir=2; end;
#          else if valeur>=1432 then do; rang=6; gir=2; end;
#        if valeur>=1432 then go to FIN;
#        /* gir D */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+0;
#            else if i=4 then valeur=valeur+0;
#            else if i=5 then valeur=valeur+2000;
#            else if i=6 then valeur=valeur+400;
#            else if i=7 then valeur=valeur+2000;
#            else if i=8 then valeur=valeur+200;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+0;
#            else if i=4 then valeur=valeur+0;
#            else if i=5 then valeur=valeur+200;
#            else if i=6 then valeur=valeur+200;
#            else if i=7 then valeur=valeur+200;
#            else if i=8 then valeur=valeur+0;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=2400 then do; rang=7; gir=2; end;
#        if valeur>=2400 then go to FIN;
#        /* gir E */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+400;
#            else if i=2 then valeur=valeur+400;
#            else if i=3 then valeur=valeur+400;
#            else if i=4 then valeur=valeur+400;
#            else if i=5 then valeur=valeur+400;
#            else if i=6 then valeur=valeur+800;
#            else if i=7 then valeur=valeur+800;
#            else if i=8 then valeur=valeur+200;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+100;
#            else if i=4 then valeur=valeur+100;
#            else if i=5 then valeur=valeur+100;
#            else if i=6 then valeur=valeur+100;
#            else if i=7 then valeur=valeur+100;
#            else if i=8 then valeur=valeur+0;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=1200 then do; rang=8; gir=3; end;
#        if valeur>=1200 then go to FIN;
#        /* gir F */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+200;
#            else if i=2 then valeur=valeur+200;
#            else if i=3 then valeur=valeur+500;
#            else if i=4 then valeur=valeur+500;
#            else if i=5 then valeur=valeur+500;
#            else if i=6 then valeur=valeur+500;
#            else if i=7 then valeur=valeur+500;
#            else if i=8 then valeur=valeur+200;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+100;
#            else if i=2 then valeur=valeur+100;
#            else if i=3 then valeur=valeur+100;
#            else if i=4 then valeur=valeur+100;
#            else if i=5 then valeur=valeur+100;
#            else if i=6 then valeur=valeur+100;
#            else if i=7 then valeur=valeur+100;
#            else if i=8 then valeur=valeur+0;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=800 then do; rang=9; gir=3; end;
#        if valeur>=800 then go to FIN;
#        /* gir G */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+150;
#            else if i=2 then valeur=valeur+150;
#            else if i=3 then valeur=valeur+300;
#            else if i=4 then valeur=valeur+300;
#            else if i=5 then valeur=valeur+500;
#            else if i=6 then valeur=valeur+500;
#            else if i=7 then valeur=valeur+400;
#            else if i=8 then valeur=valeur+200;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+200;
#            else if i=4 then valeur=valeur+200;
#            else if i=5 then valeur=valeur+200;
#            else if i=6 then valeur=valeur+200;
#            else if i=7 then valeur=valeur+200;
#            else if i=8 then valeur=valeur+100;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=650 then do; rang=10; gir=4; end;
#        if valeur>=650 then go to FIN;
#        /* gir H */
#        do i=1 to 10;
#          if i=1 then valeur=0;
#          if axei(i)='C' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+3000;
#            else if i=4 then valeur=valeur+3000;
#            else if i=5 then valeur=valeur+3000;
#            else if i=6 then valeur=valeur+3000;
#            else if i=7 then valeur=valeur+1000;
#            else if i=8 then valeur=valeur+1000;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#          else if axei(i)='B' then do;
#            if i=1 then valeur=valeur+0;
#            else if i=2 then valeur=valeur+0;
#            else if i=3 then valeur=valeur+2000;
#            else if i=4 then valeur=valeur+2000;
#            else if i=5 then valeur=valeur+2000;
#            else if i=6 then valeur=valeur+2000;
#            else if i=7 then valeur=valeur+2000;
#            else if i=8 then valeur=valeur+1000;
#            else if i=9 then valeur=valeur+0;
#            else if i=10 then valeur=valeur+0;
#          end;
#        end;
#        if valeur>=4000 then do; rang=11; gir=4; end;
#          else if valeur>=2000 then do; rang=12; gir=5; end;
#          else do; if valeur ne . then do; rang=13; gir=6; end;  else gir=.; end;
# FIN:      output;
#     run;
#  *** 16916 obs ;
#     data hid99.agirgrp (keep=ident numind gir); set agirgrp; by ident numind;
#        if enfant=1 then gir=.;
#     run;
#    /* Attention : on a mis pour les enfants de moins de 5 ans    */
#    /*  le GIR à valeur manquante, ce qui est logique             */


from collections import OrderedDict
groups = {
    'gir A': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 2000,
                2: 1200,
                3: 40,
                4: 40,
                5: 60,
                6: 100,
                7: 800,
                8: 200,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 16,
                4: 16,
                5: 20,
                6: 16,
                7: 120,
                8: 32,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 4380, 'rang': 1, 'gir': 1},
            {'valeur': 4140, 'rang': 2, 'gir': 2},
            {'valeur': 3390, 'rang': 3, 'gir': 2},
            {'valeur': 3390, 'rang': 'FIN'}
            ],
        },
    'gir B': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 1500,
                2: 1200,
                3: 40,
                4: 40,
                5: 60,
                6: 100,
                7: 800,
                8: -80,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 320,
                2: 120,
                3: 16,
                4: 16,
                5: 0,
                6: 16,
                7: 120,
                8: -40,
                9: 0,
                10: 0,
                }
            }],
        'seuils': [
            {'valeur': 2016, 'rang': 4, 'gir': 2},
            {'valeur': 2016, 'rang': 'FIN'}
            ]
        },
    'gir C': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 0,
                2: 0,
                3: 40,
                4: 40,
                5: 60,
                6: 160,
                7: 1000,
                8: 400,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 16,
                4: 16,
                5: 20,
                6: 20,
                7: 200,
                8: 40,
                9: 0,
                10: 0,
                }
            }],
        'seuils': [
            {'valeur': 1700, 'rang': 5, 'gir': 2},
            {'valeur': 1432, 'rang': 6, 'gir': 2},
            {'valeur': 1432, 'rang': 'FIN'},
            ],
        },
    'gir D': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 2000,
                6: 400,
                7: 2000,
                8: 200,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 200,
                6: 200,
                7: 200,
                8: 0,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 2400, 'rang': 7, 'gir': 2},
            {'valeur': 2400, 'rang': 'FIN'},
            ]
        },
    'gir E': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 400,
                2: 400,
                3: 400,
                4: 400,
                5: 400,
                6: 800,
                7: 800,
                8: 200,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 100,
                4: 100,
                5: 100,
                6: 100,
                7: 100,
                8: 0,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 1200, 'rang': 8, 'gir': 3},
            {'valeur': 1200, 'rang': 'FIN'},
            ]
        },
    'gir F': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 200,
                2: 200,
                3: 500,
                4: 500,
                5: 500,
                6: 500,
                7: 500,
                8: 200,
                9: 0,
                10: 0,
                }
            }, {
            'name': 'B',
            'values': {
                1: 100,
                2: 100,
                3: 100,
                4: 100,
                5: 100,
                6: 100,
                7: 100,
                8: 0,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 800, 'rang': 9, 'gir': 3},
            {'valeur': 800, 'rang': 'FIN'},
            ]
        },
    'gir G': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 150,
                2: 150,
                3: 300,
                4: 300,
                5: 500,
                6: 500,
                7: 400,
                8: 200,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 200,
                4: 200,
                5: 200,
                6: 200,
                7: 200,
                8: 100,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 650, 'rang': 10, 'gir': 4},
            {'valeur': 650, 'rang': 'FIN'},
            ],
        },
    'gir H': {
        'axes': [{
            'name': 'C',
            'values': {
                1: 0,
                2: 0,
                3: 3000,
                4: 3000,
                5: 3000,
                6: 3000,
                7: 1000,
                8: 1000,
                9: 0,
                10: 0,
                },
            }, {
            'name': 'B',
            'values': {
                1: 0,
                2: 0,
                3: 2000,
                4: 2000,
                5: 2000,
                6: 2000,
                7: 2000,
                8: 1000,
                9: 0,
                10: 0,
                },
            }],
        'seuils': [
            {'valeur': 4000, 'rang': 11, 'gir': 4},
            {'valeur': 2000, 'rang': 12, 'gir': 5},
            {'valeur': 1, 'rang': 13, 'gir': 6},
            ]
        }
    }
groups = OrderedDict(sorted(groups.items(), key=lambda t: t[0]))


if __name__ == "__main__":
    log.setLevel(logging.INFO)

    gir = get_dataframe()