import os

__author__ = 'IgorKarpov'

def main():
    postfix = '.' + 'png'
    images_names = [image_name
                    for image_name
                    in os.listdir('../brodatz_database_bd.gidx')
                    if image_name.endswith(postfix)]
    for i in images_names:
        print i

if __name__ == '__main__':
    main()