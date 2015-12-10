import pdb

from pandas import HDFStore 
import pandas.rpy.common as com     
from rpy2.robjects import r


__version__ = "0.0"

def stat(year):
    print "Calcul des statistiques individuelles"
    
    simul = "C:/til/output/simul.h5"
    simul = HDFStore(simul)
    
    df = simul['entities/register']  
    df = df.loc[df['period']==year]
    # export en R

    not_bool = df.dtypes[df.dtypes != bool]
    df = df.ix[:,not_bool.index]
    r_dataframe = com.convert_to_r_dataframe(df)
    name = 'result_sim'
    r.assign(name, r_dataframe)
    file_dir = "C:/Myliam2/output/" + name+ ".gzip"
    phrase = "save("+name+", file='" +file_dir+"', compress=TRUE)"
    r(phrase) 

    simul.close()
    
if __name__ == "__main__":
    stat(2011)