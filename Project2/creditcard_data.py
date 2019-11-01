def getData():

    cwd = os.getcwd()  #getting the path of this current program
    filename = cwd + '/default of credit card clients.xls'  #path + file


    ##For all values that are NaN, put into nanDict. Rename column name with space.
    nanDict= {}
    df = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={'default payment next month': 'defaultPaymentNextMonth'}, inplace=True)

    #Drop the rows including data where parameters are out of range
    df = df.drop(df[df.SEX<1].index)
    df = df.drop(df[df.SEX<2].index)
    df = df.drop(df[(df.EDUCATION <1)].index)
    df = df.drop(df[(df.EDUCATION >4)].index)
    df = df.drop(df[df.MARRIAGE<1].index)
    df = df.drop(df[df.MARRIAGE>3].index)

    #Features and targets
    X = df.loc[:, df.columns != 'defaultPaymentNextMonth']
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values   #returns array
    #print("typeX ",type(X))
    #print("shape X ", X.shape)
    #print("type y ", type(y))
    #print(y)


    #if one column takes in values
    onehotencoder = OneHotEncoder(categories='auto')

    #OneHot encoder for column 1,2,3 [sex,education,marriage]
    #Designmatrix
    X = ColumnTransformer(
    [('onehotencoder', onehotencoder, [1,2,3]),],
    remainder="passthrough").fit_transform(X)



    return X, np.ravel(y)
