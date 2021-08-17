<h1> This random forest regressor predicts the amount of protein bound to the protein corona, and thus gives further insight to the interactions between ENM’s (Engineered Nanomaterials) and proteins under relevant biological conditions </h1>
<p> This undergraduate research is funded by the De Novo Fellowship at Santa Clara University  </p>
<p> Written by Matthew Findlay 2017, further developed by Joseph Pham Nguyen 2021 <p>
<p> Fork me and feel free to contribute your research <p>
<h2> Implementation </h2>
<p> Here, a random forest regressor is being used as a baseline for future predictions upon predicting the amount of protein that is bound to the protein corona. <p>
<h2> Data </h2>
The experiments were setup by Danny Freitas (undergraduate bioengineering). Danny reacts nanomaterials and extracted proteins, and sends his samples to stanford where LC/MS/MS is used
To produce the spectral counts of proteins on the surface of engineered nanoparticles (bound), and proteins not on the surface of particles (unbound). Data was
mined from online databases to find the length of the proteins. The spectral counts were divided by the length of the proteins and normalized.
<h2> How to use </h2>
<p>To run the pipeline and reproduce the results call "python estimator.py (insert # of runs).
An example call might be "python estimator.py 50". This
will output the statistics and feature importances to a few output files in the “Output_Files” folder. Most of the regression information will be stored in a CSV file. If you want to visualize your data, simply type "python visualization_utils.py (input filepath)".
It'll create, display, and save the various figures and graphs for you.</p>
<h2>Custom Command-Line Arguments</h2>
<p>To make it easier to run the model for certain tasks, here is the list of specific custom CLI commands you can use.</p>
<ol>
    <li>estimator.py</li>
        <ul>
            <li>script (required): the main script that's being run (estimator.py)</li>
            <li>iterations (required): number of times to run the script</li>
            <li>input (-i, --i, argtype: filepath, optional): a file path to a desired input CSV file to make predictions on. If not provided, we make predictions on our own dataset</li>
            <li>optimize (-o, --optimize, argtype: boolean, optional): use GridSearchCV to tune RandomForestRegressor hyperparameters to optimize performance</li>
            <li>rfecv (-r, --rfecv, argtype: boolean, optional): use RFECV for feature selection to extract most useful input features</li>
            <li>example custom CLI command: "python3 estimator.py 50 -i Input_Files/example.csv" --> make predictions on example.csv</li>
            <li>for more information, simply type "python3 estimator.py -h"</li>
        </ul>
    <li>visualization_utils.py</li>
        <ul>
            <li>script (required): the main script that's being run (visualization_utils.py)</li>
            <li>input (-i, --i, argtype: filepath, optional): a file path to a desired input CSV file to make predictions on. If not provided, we make predictions on our own dataset</li>
            <li>example custom CLI command: "python3 estimator.py -i Input_Files/example.csv -o True" --> make predictions on example.csv and run GridSearchCV</li>
            <li>for more information, simply type "python3 visualizations.py -h"</li>
        </ul>
</ol>
<h2> python files </h2>
<h3> estimator.py </h3>
<p> contains main(). Runs the estimator and includes lines to run recursive feature elimination
or grid search.</p>
<h3> predictor_utils.py </h3>
<p>contains tools to build and optimize the estimator.</p>
<h3> visualization_tools.py </h3>
<p>Contains some fun functions to help us visualize the ENM data </p>
<h3>validation_utils.py</h3>
<p> Contains functions to calculate, store, and format statistics on the model performance </p>
<h3>reformat_csv.py</h3>
<p>contains various functions to perform basic CSV reformatting operations. Largely used for creating different datasets for testing.</p>
<h2> Input Files </h2>
<h3>database.csv</h3>
<p>contains the database</p>
<h3>_mask.txt</h3>
<p>A boolean mask produced by recursive feature elimination and cross validation. This mask is applied to the database to produce an optimal amount of features.</p>
<h2> Output Files </h2>
<h3> bound_fraction.png </h3>
<p>Provides a visualization of the distribution of the target variable</p>
<h3> dataset_info.txt </h3>
<p>provides basic information on the structure of our dataset</p>
<h3> dataset_numerical_attributes.csv </h3>
<p>contains more in-depth statistical information on our dataset</p>
<h3> histogram_dataset.png </h3>
<p>contains plots of the distributions from our dataset</p>
<h3>averaged_feature_importances.png</h3>
<p>a bar plot visualizing the averaged importance scores of the optimal features inputted into the model</p>
<h3> model_evaluation_info.txt </h3>
<p>provides information about averaged error metrics and feature importances for optimal features</p>
<h3>predicted_value_statistics.csv</h3>
<p>contains statistics on the predicted values generated by the model</p>
<h3>rfecv_visualization.png</h3>
<p>Visualizes the accuracy based on the number of features that are selected from the recursive feature elimination and cross validation.</p>
