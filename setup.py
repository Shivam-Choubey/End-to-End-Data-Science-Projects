from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        # 1. Read lines
        # 2. Strip whitespaces and newlines
        # 3. Filter out empty lines and the '-e .' trigger
        requirements = [req.strip() for req in file_obj.readlines()]
        
        # This filter is more robust than .remove()
        requirements = [req for req in requirements if req and not req.startswith('-e')]
            
    return requirements

setup(
    name="end_to_end_project",
    version='0.0.1',
    author="Shivam Choubey",
    author_email="shivamchubey8838@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)