import pandas as pd

# All code was run in a jupyter notebook

def LoopDetectorData(filename):
  #split vile name on _ character andextract the ID of the loop from the beginning
  loopid = filename.split('_')[0]
    
  #read Speed sheet frome excel file
  speed = pd.read_excel(filename, sheet_name='Speed')
    
  #read Volume sheet frome excel file
  volume = pd.read_excel(filename, sheet_name='Volume')
    
  #read Volume Per Lane sheet frome excel file
  volperlane = pd.read_excel(filename, sheet_name='Volume Per Lane')
    
  #read Occupancy sheet frome excel file
  occupancy = pd.read_excel(filename, sheet_name='Occupancy')

  #grab dates from the column names of the speed dataframe
  date = speed.columns[1:].to_series(index=(range(len(speed.columns[1:]))))

  #create a dataframe to received the data
  schema = {'LoopID': [], 'DateTime': [], 'Speed': [], 'Volume': []}
  res = pd.DataFrame(schema)

  #grab the range of hours of the data from the speed dataframe
  hours = speed['Unnamed: 0']
  dt = []
  ds = []
  dv = []
  id = []
  dvpl = []
  do = []

  #Loop through all available dates
  for d in date:
    j = 0
    #loop through all available hours and grab data from dataframe locations matching unique date and times
    for hour in hours:
      id.append(loopid)
      dt.append(d + ' ' + hour)
      ds.append(speed.loc[j, d])
      dv.append(volume.loc[j, d])
      dvpl.append(volperlane.loc[j, d])
      do.append(occupancy.loc[j, d])
      j += 1

  #merge data in dataframe
  res['DateTime'] = pd.Series(data=dt)
  res['Speed'] = pd.Series(data=ds)
  res['Volume'] = pd.Series(data=dv)
  res['Volume Per Lane'] = pd.Series(data=dvpl)
  res['Occupancy'] = pd.Series(data=do)
  res['LoopID'] = pd.Series(data=id)

  #name and save an output csv file
  outputfilename = loopid + '_loop_cloutput.csv'
  res.to_csv(outputfilename)
  return outputfilename

#call for both I-5 and 520 loop data files
filename_i5 = LoopDetectorData('005es16732_MS___1_MoTuWeThFr_2015-01-01_2015-12-31.xlsx')
filename_520 = LoopDetectorData('520es00972_MW___3_MoTuWeThFr_2015-01-01_2015-12-31.xlsx')