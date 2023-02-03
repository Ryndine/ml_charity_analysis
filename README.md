# Neural Networks - Charity Analysis

## Objective
Create a binary classifier that is capable of predicting whether applicants will be successful if they recieve funding.

## Tools
* Python
* Pandas
* Scikit-Learn
* Tensorflow

## Exploration & Cleanup
```
print(loan_stats_df.shape)
(34299, 10)
print(loan_stats_df.isnull().values.any())
False
```
First step I always check the shape and whether the data has any nulls. I see my dataset is fairly average size and does not contain any nulls.

```
application_df = application_df.drop(["EIN","NAME"],axis=1)
```
Next I see I have identification numbers and company names. Since my goal for this machine learning is to create a model that can predict whether applicants are successfull, I know these two columns are not needed for the machine learning, so I'm dropping them.

```
application_df.nunique()
APPLICATION_TYPE            17
AFFILIATION                  6
CLASSIFICATION              71
USE_CASE                     5
ORGANIZATION                 4
STATUS                       2
INCOME_AMT                   9
SPECIAL_CONSIDERATIONS       2
ASK_AMT                   8747
IS_SUCCESSFUL                2
```
I need to inspect the dataframe for potential binning. I see ASK_AMT has a lot of values which is to be expected but I don't want to bin those. They're likely to be important to the overall machine learning accuracy. "APPLICATION_TYPE" and "CLASSIFICATION" stand out as two columns that can potentially be binned.

```
application_counts = application_df['APPLICATION_TYPE'].value_counts()
application_counts.plot.density()

replace_application = list(application_counts[application_counts < 500].index)
for app_id in replace_application:
    application_df['APPLICATION_TYPE'] = application_df['APPLICATION_TYPE'].replace(app_id,"Other")
application_df['APPLICATION_TYPE'].value_counts()
```
Looking further into application_type I see I do have an opportunity to bin. If I bin everything below 500 that will clean up all of the small values and help reduce noise for the prediction model.

```
class_counts = application_df['CLASSIFICATION'].value_counts()
class_counts.plot.density()
replace_classification = list(class_counts[class_counts < 1000].index)
for class_id in replace_classification:
    application_df['CLASSIFICATION'] = application_df['CLASSIFICATION'].replace(class_id,"Other")
application_df['CLASSIFICATION'].value_counts()
```
Next I need to look at classifcation. As with the previous column I see a lot of noise in the "under 1000" range. So I'm binning those values.

```
application_cat = application_df.dtypes[application_df.dtypes == "object"].index.tolist()
```
Since I'll be using OneHotEncoder, I'm setting aside a list of my categories for later. I want this to simplify the upcoming code.

```
enc = OneHotEncoder(sparse=False)
encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))
encode_df.columns = enc.get_feature_names(application_cat)
```
I'm using my category list here to fit and transform the OneHotEncoder and add the encoded variable names to the dataframe.

```
application_df = application_df.merge(encode_df,left_index=True, right_index=True)
application_df = application_df.drop(application_cat, axis=1)
```
To finish the OHE I need to merge the OHE features and drop the originals. From here I'm ready to train/test.


## Machine Learning: Train Test Split
```
y = application_df["IS_SUCCESSFUL"].values
X = application_df.drop(["IS_SUCCESSFUL"], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
```
My goal is to predict whether a charity applicant will be successful, my target is the "IS_SUCCESSFUL" column. To start I'm putting in the rest of my columns as features, but will adjust upon further evaluation during machine learning.

```
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```
Next I'll be applying the StandardScaler to my dataset in order to normalize my data.

## Machine Learning: Compile, Train, Evaluate
```
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  40
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.InputLayer(input_shape=(43)))
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
nn.summary()
```
With my deep neural network, I'm giving it an input equivalent to how many features I have. For my hidden nodes I'm starting with between the size of my input features and my output layer, so 40. I'm doing a single hidden layer since adding more becomes more difficult to train, and with this data a single layer should be enough. Since I'm not going to be using softmax on the output layer, I'm keeping the units to 1.

```
os.makedirs("charity_checkpoints/",exist_ok=True)
checkpoint_path = "charity_checkpoints/weights.{epoch:02d}.hdf5"
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')
fit_model = nn.fit(X_train_scaled, y_train, epochs=100, callbacks=[cp_callback])
```
Next I'm going to be setting a callback to save the models weights. Then fit the data to the neural network.

```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 0.5534 - accuracy: 0.7259 - 529ms/epoch - 2ms/step
Loss: 0.5534329414367676, Accuracy: 0.7259474992752075

nn.save("charity.h5")
```
My first attempt didn't return that great of results. I have an okay accuracy but the loss is way too high. Moving forward I'll look to improve it.

## Second Attempt
```
name_counts = application_df['NAME'].value_counts()
name_counts[name_counts>5]
name_counts.plot.density()
```
For my second attempt, I'm thinking of adding the names column back into the dataset. Inspecting it for binning, and deciding names may be worth keeping for our data. Everything else I'll be keeping the same.

```
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  263
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.InputLayer(input_shape=(43)))
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
nn.summary()
```
With my new data I went from 43 inputs to 398. So, I'll be setting my hidden nodes first, to about 2/3 of my input layer + output layer. Everything I'll keep the same, then look into tuning it again after.

```
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - loss: 0.4896 - accuracy: 0.7911 - 566ms/epoch - 2ms/step
Loss: 0.4895803928375244, Accuracy: 0.7911370396614075
```
Looking at results I see I've improved the neural network. Next I'm going to attempt looking at the hidden nodes.

```
Ni = number of input neurons.
No = number of output neurons.
Ns = number of samples in training data set.
α = an arbitrary scaling factor usually 2-10.
Nh = Ns / (α∗(Ni+No))

268/268 - 0s - loss: 0.4535 - accuracy: 0.7908 - 465ms/epoch - 2ms/step
Loss: 0.45353230834007263, Accuracy: 0.7907871603965759
```
I'll be using this formula to test a new number of neurons, using 2 as my scaling factor. This comes out to about 33. After running my test my model seemed to perform a bit better with less loss.

```
hidden_nodes_layer1 = 263
hidden_nodes_layer2 = 33
# first hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)
# second hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="sigmoid")
)

268/268 - 0s - loss: 0.4458 - accuracy: 0.7889 - 370ms/epoch - 1ms/step
Loss: 0.4457626938819885, Accuracy: 0.7889212965965271
```
Next I'm going to test a second layer set to sigmoid since I don't want two relu layers. Since ReLU may result in "dying ReLU" I want to avoid have two of them. Since sigmoid is more computationally expensive I'll set that to the smaller layer. I seem to be losing accuracy, but improving loss.

```
# first hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="sigmoid")
)
# second hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu")
)

268/268 - 1s - loss: 0.4667 - accuracy: 0.7901 - 558ms/epoch - 2ms/step
Loss: 0.46665477752685547, Accuracy: 0.7900874614715576
```
My last attempt for this project will be setting the first layer to sigmoid, and the second to relu. As previously stated, factoring in "dying ReLU" I decided it may be better to set that to the second layer. This is in hopes that less data would end up dying. However, after running the model I see my loss went back up which indicates ReLU is the better layer to start with.

```
rf_model = RandomForestClassifier(n_estimators=128, random_state=78) 
rf_model = rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print(f" Random forest model accuracy: {accuracy_score(y_test,y_pred):.3f}")

Random forest model accuracy: 0.776
```
The last thing I want to test is a simple RandomForestClassifier to see if the neural network was worth it.

Since my results didn't yield significant improvements over the RFC, I can't say it was worth it. The computational requirement of the neural network relative to the accuracy didn't justify using it.