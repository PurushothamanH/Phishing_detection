from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this fun returns requirements as list
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        
        ## used to filter or remove \n from requirements list outcome.
        requirements=[req.replace("\n", " ") for req in requirements]

        ## this below cmd used to remove effect of -e . while doing the function
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements        
        ## -e . used in requirements file is for connecting the requirement file with setup file..

setup(
name="Phishing_detection",
version='0.0.1',
author='purushothaman',
author_email='haripurushoth185@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)