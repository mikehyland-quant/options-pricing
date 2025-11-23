#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports
import pandas as pd
import xlwings as xw


# In[ ]:


#these are quick, self built utility methods used in almost all of the self-written IBKR code
    
def getDict(workbook, worksheet, cell, style=float):  
    wb = xw.Book(workbook)    
    mySheet = wb.sheets[worksheet]
    myRange = mySheet.range(cell).value
    tempDict = mySheet.range(myRange).options(dict, numbers=style).value
    return tempDict

def getDF(workbook, worksheet, myRange, headerRows=1, style=float):    
    wb = xw.Book(workbook)
    mySheet = wb.sheets[worksheet]
    tempDF = mySheet.range(myRange).options(pd.DataFrame, index = False, numbers=style, header=int(headerRows)).value
    return tempDF

def dictToDF(dictionary):
    dataFrameData = []
    for key, obj in dictionary.items():
        tempDict = obj.__dict__
        dataFrameData.append(tempDict)
    df = pd.DataFrame(dataFrameData)
    return df
            
def printDFToXL(workbook, worksheet, xlRange, df, cols=[], headerRows=1):    
    if cols != []:
        df = df[cols]
    wb = xw.Book(workbook)    
    mySheet = wb.sheets[worksheet] 
    mySheet.range(xlRange).options(index = False, header=headerRows).value = df
    
def printDictToXL(dictionary, printDetails): 
    attrs = printDetails[0]
    sortCols = printDetails[1]
    printInfo = printDetails[2]
    df = dictToDF(dictionary)
    df.sort_values(sortCols, inplace=True)
    printDFToXL(printInfo[0], printInfo[1], printInfo[2], df, cols=attrs)

def flatten_dict_column(df, column_name, sep='_'):
    """
    Flatten a column of dictionaries (including nested dicts) into separate columns.

    Args:
        df (pd.DataFrame): original DataFrame
        column_name (str): name of the column containing dicts
        sep (str): separator for nested keys

    Returns:
        pd.DataFrame: original DataFrame joined with flattened dict columns
    """
    # Use json_normalize for nested dicts
    flattened = pd.json_normalize(df[column_name], sep=sep)
    
    # Optionally prefix with original column name to avoid conflicts
    flattened.columns = [f"{column_name}{sep}{col}" for col in flattened.columns]
    
    # Join to original DataFrame
    df = df.drop(columns=[column_name]).join(flattened)
    return df

