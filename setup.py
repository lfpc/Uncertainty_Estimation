from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='NN_Torch',
    url='https://github.com/lfpc/NN_torch',
    author='Luís FP Cattelan',
    author_email='luisfelipe1998@gmail.com',
    # Needed to actually package something
    packages=find_packages(where='src'),
    # Needed for dependencies
    install_requires=['numpy','torch'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Neural Networks in Pytorch - utils and losses, models e uncertainty quantifications definitions',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)