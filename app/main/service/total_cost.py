import math
def totalcost(df_error):
    # calculated cost function
    result = 0
    for i in range(len(df_error)):
        if df_error['total_cost'].iloc[i] < 0:
            calculated = math.exp((df_error['total_cost'].iloc[i])/-13)-1
            result += calculated
        else:
            calculated = math.exp((df_error['total_cost'].iloc[i])/10)-1
            result += calculated
    Total_cost = result/len(df_error)
    return int(Total_cost)