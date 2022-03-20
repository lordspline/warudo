"""emergency_dataset dataset."""

import tensorflow_datasets as tfds
import os
import numpy as np
import tensorflow as tf

# TODO(emergency_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(emergency_dataset): BibTeX citation
_CITATION = """
"""


class EmergencyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for emergency_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(emergency_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(512, 512, 3)),
            'label': tfds.features.Tensor(shape=(100, 5), dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(emergency_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(emergency_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples("train"),
        'val' : self._generate_examples("val"),
    }

  def _generate_examples(self, split):
    """Yields examples."""
    # TODO(emergency_dataset): Yields (key, example) tuples from the dataset
    
    if split == "train":
        _range = range(700)
    else:
        _range = range(700, 755)
    
    for i in _range:
        
        if i in [592, 593, 594, 595]:
            continue
        
        labelfile = open(os.path.join("./newData/labels", str(i+1) + ".txt"), 'rb')
        label = np.loadtxt(labelfile, delimiter=' ', dtype=np.float32)
        label = np.reshape(label, (100, 5))
        # label = np.concatenate((label[1:], np.array([1,0])))
        # label = tf.convert_to_tensor(label, dtype=tf.float32)
        labelfile.close()
    
        yield "record" + str(i+1), {
            'image': os.path.join("./newData/images", str(i+1) + ".jpg"),
            'label': label,
        }
