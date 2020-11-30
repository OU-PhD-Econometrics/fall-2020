using MLJ, Tables, DataFrames, MLJDecisionTreeInterface, MLJLinearModels

models()

# 151-element Array{NamedTuple{(:name, :package_name, :is_supervised, :docstring, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :is_pure_julia, :is_wrapper, :load_path, :package_license, :package_url, :package_uuid, :prediction_type, :supports_online, :supports_weights, :input_scitype, :target_scitype, :output_scitype),T} where T<:Tuple,1}:
#  (name = ARDRegressor, package_name = ScikitLearn, ... )
#  (name = AdaBoostClassifier, package_name = ScikitLearn, ... )
#  (name = AdaBoostRegressor, package_name = ScikitLearn, ... )
#  ⋮
#  (name = XGBoostClassifier, package_name = XGBoost, ... )
#  (name = XGBoostCount, package_name = XGBoost, ... )
#  (name = XGBoostRegressor, package_name = XGBoost, ... )

# use house price data from US Census Bureau
df = OpenML.load(574) |> DataFrame
X = df[:,[:P1,:P5p1,:P6p2,:P11p4,:P14p9,:P15p1,:P15p3,:P16p2,:P18p2,:P27p4,:H2p2,:H8p2,:H10p1,:H13p1,:H18pA,:H40p4]]
X = Tables.rowtable(X)
y = log.(df.price)

models(matching(X,y))

# declare a tree and lasso model
tree_model = @load DecisionTreeRegressor pkg=DecisionTree
lasso_model = @load LassoRegressor pkg=MLJLinearModels

# initialize "machines" where results can be reported
tree = machine(tree_model, X, y)
lass = machine(lasso_model, X, y)

# split into training and testing data
train, test = partition(eachindex(y), 0.7, shuffle=true)

# train the models
MLJ.fit!(tree, rows=train)
MLJ.fit!(lass, rows=train)

# predict in test set
yhat = MLJ.predict(tree, X[test,:]);
yhat = MLJ.predict(lass, X[test,:]);

# get RMSE across validation folds
MLJ.evaluate(tree_model,  X, y, resampling=CV(nfolds=6, shuffle=true), measure=rmse)
MLJ.evaluate(lasso_model, X, y, resampling=CV(nfolds=6, shuffle=true), measure=rmse)
