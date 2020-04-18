# from transformers import InputExample
from utils import LinkingExample
from transformers import DataProcessor
import os
import logging


# 采用InputExample对数据进行封装
# DataSetProcessor用于处理读入recall阶段的数据


logger = logging.getLogger(__name__)


class DataSetProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return LinkingExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "military_training.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "military_training.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        label_list = []
        for i in range(104520):
            label_list.append(str(i))
        return label_list

    # 修改读取函数，把mention_id也读取进来，方便ranker查找对应的mention
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            label = line[0]
            mention_id = line[1]  # 增加对于mention_id的读取，以便在ranking阶段读入mention text
            examples.append(LinkingExample(guid=guid, text_a=text_a, label=label, mention_id=mention_id))
        return examples

