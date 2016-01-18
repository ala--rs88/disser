import operator

__author__ = 'IgorKarpov'


class kNNClassifier:

    k = -1
    finder = None

    def __init__(self, k, finder):
        self.k = k
        self.finder = finder

    def learn(self, train_data_source, params_dict):
        self.finder.learn(train_data_source, params_dict)

    def classify_image(self, query_image):
        top_matching_images_indexes = self.finder.find_top_matches(query_image,  self.k)
        top_matching_classes = [self.finder.data_source.get_image_class(image_id)
                                for image_id
                                in top_matching_images_indexes]
        top_class = kNNClassifier.__get_prevailing_values(top_matching_classes)
        return top_class

    @staticmethod
    def __get_prevailing_values(values):
        occurrences = {}
        for value in values:
            if value in occurrences:
                occurrences[value] += 1
            else:
                occurrences[value] = 1

        prevailing_value = max(occurrences.iteritems(), key=operator.itemgetter(1))[0]
        return prevailing_value
