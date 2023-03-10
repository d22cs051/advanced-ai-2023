import plotly.graph_objects as go # Install using `pip3 install plotly`

def Plot(input_series: list, plot_fn_output: list):
    l = len(input_series)
    m = len(plot_fn_output)
    fig = go.Figure()
    gmode = 'lines+markers'
    fig.add_trace(go.Scatter(x=list(range(0, l)), y=input_series,
                  mode=gmode, name='Original series'))
    fig.add_trace(go.Scatter(x=list(range(l-1, l+m+1)),
                  y=plot_fn_output, mode=gmode, name='Predicted'))
    fig.show()

S1 = [
    62.28040320585449, 58.59711419069721, 79.82303965145344, 99.31686271024961, 106.38882835649932,
    88.17619704646604, 84.49290803130877, 105.71883349206499, 125.21265655086115, 132.28462219711088,
    114.07199088707758, 110.3887018719203, 131.61462733267652, 151.10845039147267, 158.1804160377224,
    139.96778472768915, 136.28449571253185, 157.51042117328808, 177.00424423208426, 184.07620987833394,
    165.86357856830068, 162.18028955314338, 183.40621501389964, 202.9000380726958, 209.97200371894553,
    191.75937240891224, 188.07608339375494, 209.30200885451117, 228.79583191330732, 235.86779755955706,
    217.65516624952377, 213.97187723436647, 235.1978026951227, 254.6916257539189, 261.7635914001686,
    243.55096009013533, 239.86767107497803, 261.09359653573426, 280.58741959453045, 287.6593852407801,
    269.44675393074687, 265.76346491558957, 286.9893903763458, 306.483213435142, 313.55517908139166
]

S2 = [
    20.90726190744447, 35.160881753959686, 15.418706781228899, 27.775710763062012, 32.72073606917471,
    27.794994549582217, 25.212928819264356, 34.29926902641212, 11.733543497702057, 21.50867692357205,
    34.93253990887256, 28.507681392705283, 24.80981135607093, 35.814112028468266, 10.29510053587015,
    23.72822780179776, 37.982477843443895, 23.203960926234092, 24.289317088661452, 32.297119253784054,
    11.67465693651977, 21.650505994225746, 33.818301679559625, 22.780292894454746, 21.317248166657926,
    30.41331325610611, 14.986069524100227, 26.757096808096378, 32.84024374300274, 24.036707860493973,
    20.895256565839475, 32.70525332389984, 15.872651125289975, 29.162644721112706, 36.68629742572546,
    22.118930495716548, 23.091077202332915, 37.95974067529713, 9.422372770044024, 21.150275919812852,
    35.77902415288387, 25.463892468411014, 21.534214777303283, 31.67300638275717, 13.065448758118368,
    24.141364641128245, 38.5095130095619, 20.417196728966076, 25.336104085668595, 39.21477510428962,
    9.882404539492503, 26.072661949954647, 37.58455797127314, 26.551531200137745, 22.494033061906133,
    35.43175251664542, 12.72262628500311, 23.77265231866362, 36.43695246941256, 27.00842776715278
]

S3 = [
    117.9750865407774, 120.49631021427636, 127.25465991516671, 129.47373128366218, 127.79632344265902,
    115.40091919127902, 107.47368062707788, 120.42385750216656, 113.40402328097605, 122.7589675738068,
    123.76912541938268, 126.16688917804792, 135.55247463468987, 123.61293239771899, 98.35931797222968,
    118.84719912714522, 126.56670626926511, 131.62135372549548, 124.06614121728042, 137.05074620483074,
    127.27994655190034, 120.69519339536416, 103.68302632180217, 113.07336149748494, 121.57862336152104,
    122.00968756888278, 124.00106908267382, 131.53465424838294, 138.58613595196454, 115.88616775837444,
    102.47807903317033, 122.38806671745034, 114.35041510290544, 125.01765102132293, 122.40464205036707,
    125.74206174709924, 137.57697780738684, 109.56080365749615, 101.43252715830332, 114.75849695193351,
    122.75853433817512, 130.3746607652826, 119.9638649865455, 121.11886832389494, 136.5566059640904,
    113.02410052639097, 103.01847824353698, 108.8282249439293, 125.1876875242933, 117.38399927806667,
    120.57762723156542, 124.41523179023103, 134.53643644885784, 120.99213760622963, 103.68027928132341, 107.19182610129685
]

S4 = [
    10, 18.767049999999998, 14.687613975000001, 33.1585559442625, 37.90687153708975, 17.921317914507082,
    37.8688418140963, 32.463637442549, 52.79663965194792, 54.26639244257362, 31.977677774727095, 51.56112144430142,
    46.0934620611588, 66.05303097871341, 67.90804555645121, 49.0297441670741, 68.61318783664842, 63.145528453505804,
    83.10509737106041, 84.96011194879821, 66.0818105594211, 85.66525422899542, 80.19759484585282, 100.15716376340741,
    102.01217834114524, 83.1338769517681, 102.71732062134242, 97.24966123819982, 117.20923015575441, 119.06424473349225,
    100.18594334411512, 119.76938701368942, 114.30172763054682, 134.26129654810143, 136.11631112583925, 117.23800973646213,
    136.82145340603643, 131.35379402289382, 151.31336294044843, 153.16837751818625, 134.29007612880915, 153.87351979838343,
    148.40586041524085, 168.36542933279543, 170.22044391053325, 151.34214252115615, 170.92558619073043, 165.45792680758782,
    185.41749572514243, 187.27251030288025, 168.39420891350315, 187.97765258307746, 182.50999319993485, 202.46956211748943,
    204.32457669522725, 185.44627530585015, 205.02971897542446, 199.56205959228186, 219.52162850983646, 221.37664308757425,
    202.49834169819715, 222.08178536777146, 216.61412598462886, 236.57369490218346, 238.42870947992128, 219.55040809054415,
    239.13385176011846, 233.66619237697586, 253.62576129453046, 255.48077587226828, 236.60247448289118, 256.1859181524655,
    250.71825876932286, 270.67782768687744, 272.5328422646153, 253.6545408752382, 273.2379845448125, 267.77032516166986,
    287.72989407922444, 289.5849086569623, 270.7066072675852, 290.2900509371595, 284.82239155401686, 304.78196047157144,
    306.6369750493093, 287.75867365993213, 307.3421173295065, 301.87445794636386, 321.8340268639185, 323.6890414416563
]

S5 = [
    1122.244203975118, 708.2668562415706, 1198.5783669127334, 1403.1309961555833,
    1211.4498327291237, 1379.8883786862839, 1108.2462852960027, 645.8697222258771,
    1255.6305876514898, 1410.6567977713732, 1250.9380178434874, 1317.7042771955093,
    994.4815406758429, 785.9299384908829, 1229.0751267002988, 1427.3589866374116,
    1239.4320263743623, 1317.9338606183967, 954.915975819304, 733.3931099640309,
    1188.4807592160935, 1397.5880298768625, 1246.7563097065565, 1345.2628305433718,
    1098.6667703167725, 717.285495641424, 1188.3407433576515, 1308.805033177505,
    1412.243318506125, 1390.9797807774994, 1097.4587548191835, 692.0334756855785, 
    1249.8362396298633, 1309.9208466887744, 1379.283570406405, 1417.772482519472, 
    966.0947259407172, 722.7450856121811, 1176.9237430323121, 1409.0718239260582, 
    1303.1508748077817, 1470.2484014275208, 1102.030828304322, 640.4815552078887, 
    1272.0218675419255, 1386.0037321004188, 1360.6596092424004, 1335.8325643969754
]

import forecasting

if __name__ == "__main__":

    ARIMA_Sample = [ # A series to test the arima forecasting
        6.988045118523939, 16.936183972662768, 27.894793932851556, 11.360123882026187, 
        6.796995800146647, 18.30869955623013, 34.51655112738614, 17.427290748999486, 
        14.758830109387528, 17.45345650377549, 34.32732873976919, 18.836214738732913, 
        11.221346213169166, 14.796636160332834, 27.559055952243373, 15.298984194488494, 
        13.564500020803997, 15.854546559775184, 25.6623452565195, 10.116605931110058
    ]

    ARIMA_Forecast = forecasting.ARIMA_Forecast(ARIMA_Sample, 2, 2, 5, 15) 
    Plot(ARIMA_Sample, ARIMA_Forecast)
    # View the plot for the forecast

    
    HoltWinters_Sample = [ # A series to test the Holt-Winters forecasting
        70, 76.60000000000001, 55.32200000000001, 37.885740000000006, 49.9868458, 
        85.125464686, 95.88600759961999, 71.91134568594542, 49.053790529697416, 58.84815205311365, 
        79.66504402709089, 85.68802902863068, 65.71096388270314, 45.61979921835485, 54.835723887304894, 
        74.64950881982993, 80.67249382136974, 60.69542867544219, 40.60426401109391, 49.82018868004395, 
        69.633973612569, 75.6569586141088, 55.67989346818124, 35.58872880383297, 44.80465347278301, 
        64.61843840530805, 70.64142340684786, 50.6643582609203, 30.573193596572033, 39.78911826552206, 
        59.60290319804711, 65.62588819958692, 45.64882305365936, 25.557658389311086, 34.773583058261124
    ]
    HoltWinters_Forecast = forecasting.HoltWinter_Forecast(HoltWinters_Sample, 0.7, 0.9, 0.4, 5, 25)
    Plot(HoltWinters_Sample, HoltWinters_Forecast)
    # View the plot for the forecast
    
    series = [S1,S2,S3,S4,S5]
    for series in series:
        # for arima model
        (P, D, Q), ARIMA = forecasting.ARIMA_Paramters(series) # getting best parms and preds
        Plot(series, ARIMA)
        
        # for holt winters
        (Alpha, Beta, Gamma, Seasonality),HoltWinters = forecasting.HoltWinter_Parameters(series) # getting best parms and preds
        Plot(series, HoltWinters)
        
