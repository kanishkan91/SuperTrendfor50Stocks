import glob
import pandas as pd
import xlsxwriter


#Update path below
path = (r'Nifty 50 Companies Merged')

filenames = glob.glob(path + "/*.csv")
print("Reading files from path" + str(path))

data= []
for filename in filenames:
    filename = pd.read_csv(filename)
        #filename = pd.merge(filename, CountryConcord, how='left', left_on='location_name',
                            #right_on='Country name in IHME')
        #filename = pd.merge(filename, SeriesConcord, how='left', left_on='cause_name', right_on='Series name in IHME')
    filename = filename.dropna(how='any')
        # GBDDalys.append(pd.read_csv(filename,low_memory=False))
    data.append(filename)

data = pd.concat(data, ignore_index=True)

#data=pd.read_excel('ProjectUdaan.xlsx')
data9=pd.DataFrame(data)

writer = pd.ExcelWriter('ConsolidatedData.xlsx', engine='xlsxwriter')
data9.to_excel(writer, sheet_name='Revenue', merge_cells=False)

writer.save()