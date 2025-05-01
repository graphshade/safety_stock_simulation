from dotenv import load_dotenv
import sys
import pandas as pd

import safety_stock_utils
from logger import logger

if __name__ == "__main__":

    #load environment variables if any
    load_dotenv()

    #data prep : get consumption input data
    try:
        # data preparation code goes here (dummy data)
        df = pd.DataFrame([{"itemID":["item1", "item2"],
                            "qty":[234,479]}])
        pass
        
    except Exception as error:
        logger.critical(error, stack_info=True)
    
    ### Safety stock run
    try:
        #Fit custom distributions
        #Loop over each unique item and fit custom distribution and save the result as a
        # list of dictinaries of unique items and fitted demand distribution with probability mass function
        kde_li = []
        for item in [df["itemID"].unique()]:
            try:
                d_x, d_pmf = safety_stock_utils.fit_discrete_kde(df[df['itemID']==item]['qty'])
                kde_li.append({"itemID":item, "d_x":d_x,"d_pmf":d_pmf})
            #for itemID where there is lower subspace in data in order to fit custom distribution,skip
            except Exception as error:
                pass
        #generate dataframe from results and merge to original dataset
        df_kde = df.merge(pd.DataFrame(kde_li),
                        on="itemID",
                        how="left")
        #remove instances of 'd_x' being NaN
        df_kde = df_kde[~df_kde['d_x'].isna()]
        
        logger.info("custom fitting successful")

        ### Run Simulation
        df_kde['result'] =  df_kde.apply(lambda x: safety_stock_utils.simulate_safety_custom(d_x=x['d_x'], 
                                                                                                d_pmf=x['d_pmf'],
                                                                                                pu_price=x['price'],
                                                                                                R=x['R_value'],
                                                                                                L=x['lt_wk'],
                                                                                                alpha=x['SL_value'], 
                                                                                                time=100000),
                                            axis=1)
        df_kde[["t_alpha", "sl_alpha", "sl_period", "ss", "ss_value"]] = pd.DataFrame(df_kde['result'].tolist(),index=df_kde.index)
        df_kde.drop(columns=['result', 'd_x','d_pmf'],inplace=True)
        df_kde.to_csv(f"./results/outputs_kde/ss_out_.csv", index=False)
        
        logger.info("simulation completed") #must delete
    except Exception as error:
        logger.critical(error, stack_info=True)
        sys.exit(1)
 