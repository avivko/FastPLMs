## Creating a new Sigularity image

To create a new version, once must use a non-Sherlock machine. This is because Sherlock does not give root access, which is needed to build intermediate Docker images. To release a new software version: tag the repo with the version, run make, and then copy the output to our group space. Versioning uses the CalVer YYYY.MM.MICRO syntax: year, month, and monthly release number.

Example release, on a non-Sherlock machine (modify as needed):
```
cd <path_to_repo>/FastPLMs

REMOTE_USER=kormanav
TAG_DATE=2026.04.0 # YYYY.MM.MICRO change accordingly
IMAGE_NAME=fastplms
SIF_REMOT_DEST=/oak/stanford/groups/rondror/software/plms/singularity

SIF_NAME=${IMAGE_NAME}_${TAG_DATE}.sif

git tag $TAG_DATE 
make
scp /tmp/${IMAGE_NAME}/${SIF_NAME} ${REMOTE_USER}@dtn.sherlock.stanford.edu:${SIF_REMOT_DEST}
```
To make the new version the default, update the symlink on a Sherlock machine (adjust according to the parameters you used):
```
# cd to SIF_REMOT_DEST
cd oak/stanford/groups/rondror/software/plms/singularity
# link tagged version (SIF_NAME) as the new main sif (IMAGE_NAME.sif)
ln -sf fastplms_2026.04.0.sif fastplms.sif
```