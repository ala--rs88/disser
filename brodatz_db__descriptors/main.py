from evaluator import Evaluator


__author__ = 'IgorKarpov'


def main():
    print('main started')

    evaluator = Evaluator('brodatz_database_bd.gidx')
    accuracy = evaluator.evaluate_accuracy()
    print 'accuracy = %f', accuracy


if __name__ == '__main__':
    main()