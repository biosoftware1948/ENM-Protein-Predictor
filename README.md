<h1> This Script Predicts Interactions between engineered nanomaterials and proteins under relevant biological conditions </h1>
<p> This is funded undergraduate research from the Dr. Wheeler lab at Santa Clara University </p>
<h2> Implementation </h2>
<p> This script uses a random forest classifier to make predictions <p>
<h2> Data </h2>
The experiments were setup by Danny Freitas (undergraduate bioengineering). Danny reacts nanomaterials and extracted proteins, and sends his samples to stanford where LC/MS/MS is used
To produce the spectral counts of proteins on the surface of engineered nanoparticles (bound), and proteins not on the surface of particles (unbound). Data was
mined from online databases to find the length of the proteins. The spectral counts were divided by the length of the proteins,
normalized, and a ratio was created giving a NSAF value representing the enrichment of specific proteins onto the surface of nanomaterials.
With this enrichment factor, databases were mined for protein characteristics, and this information was used to predict if a given protein
and nanomaterial pair will bind
