<h1> This Script Predicts Interactions between engineered nanomaterials and proteins under relevant biological conditions </h1>
<p> This is funded undergraduate research from the Dr. Wheeler lab at Santa Clara University </p>
<p> Written by Matthew Findlay 2017 <p>
<p> Fork me and feel free to contribute your research <p>
<h2> Implementation </h2>
<p> This script uses a random forest classifier to make predictions. Several different machine learning algorithms and ensembles were used but we ultimately decided to stick with random forests due to their human readability <p>
<h2> Data </h2>
The experiments were setup by Danny Freitas (undergraduate bioengineering). Danny reacts nanomaterials and extracted proteins, and sends his samples to stanford where LC/MS/MS is used
To produce the spectral counts of proteins on the surface of engineered nanoparticles (bound), and proteins not on the surface of particles (unbound). Data was
mined from online databases to find the length of the proteins. The spectral counts were divided by the length of the proteins,
normalized, and a ratio was created giving a NSAF value representing the enrichment of specific proteins onto the surface of nanomaterials.
With this enrichment factor, databases were mined for protein characteristics, and this information was used to predict if a given protein
and nanomaterial pair will bind
<h2> How to use </h2>
<p>To run the pipeline and reproduce the results call estimator.py "amount of runs" "output file" (2 command line arguments). This
will output the statistics and feature importances to the "output file" in JSON format. This will also
output all the classification information to a csv file. To see the statistics and feature importances
in a readable format call statistic_parser.py "output_file" and the results will be printed to the
command line.</p>
<h2> Predicting your own data</h2>
<p> To make predictions of your own data go to the main function in estimator.py. Set db.predict = "your_csv_path". This will use our data to make predictions on yours. A csv file will be outputted
with easily interpretable results. Be weary of predictions that fall within the 0.4-0.6 probability range
as these are considered unreliable </p>
<h2> python files </h2>
<h3> estimator.py </h3>
<p> contains main(). Runs the estimator and includes lines to run recursive feature elimination
or grid search.</p>
<h3> predictor_utils.py </h3>
<p>contains tools to build and optimize the estimator.</p>
<h3> visualization_tools.py </h3>
<p>Contains some fun functions to help us visualize the ENM data </p>
<h3> validator.py </h3>
<p> Contains functions and classes that make it easy to validate the model performance </p>
<h3>statistic_parser.py</h3>
<p> parses the model JSON output and prints it in readable format to the command line </p>
<h2> Input Files </h2>
<h3>database.csv</h3>
<p> contains the database </p>
<h3>mask.txt</h3>
<p> a boolean mask produced by recursive feature elimination and cross validation.
This mask is applied to the database to produce an optimal amount of features </p>
<h2> Output Files </h2>
<h3> final.json </h3>
<p>non stratified results from running the model 50x</p>
<h3>stratified_final.json</h3>
<p>stratified results from running the model 50x</p>
<h3>prediction_probability.csv</h3>
<p>contains an excel file with information about predictions</p>
<h3>stratified_prediction_probability.csv</h3>
<p>contains an excel file with information about stratified predictions</p>
<h3> y_randomization.csv</h3>
<p>results from the y_randomization_test</p>
