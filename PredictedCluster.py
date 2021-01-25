
from sklearn.cluster import KMeans
import shutil,glob,os
import TFidf as tf

dir_path="resume_samples"
glob_dir = dir_path + '/*'
paths = [file for file in glob.glob(glob_dir)]

k=6
km = KMeans(n_clusters=k, random_state=10)
kpredictions=km.fit_predict(tf.denselist)
print("kpredictions",kpredictions)
shutil.rmtree('resume-output')

for i in range(k):
    os.makedirs("resume-output\cluster" + str(i))

for i in range(len(paths)):
    shutil.copy2(paths[i], "resume-output\cluster"+str(kpredictions[i]))