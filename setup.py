from setuptools import setup

setup(
    name='neural-net-from-scratch',
    version='1.0',
    description='A simple implementation of a neural network from scratch in Python',
    long_description='A simple implementation of a neural network from scratch in Python',
    long_description_content_type='text/markdown',
    author='Samy Alderson',
    author_email='samy.alderson@example.com',
    url='https://github.com/samy-alderson/neural-net-from-scratch',
    packages=['src', 'tests'],
    package_dir={'src': 'src', 'tests': 'tests'},
    include_package_data=True,
    install_requires=['numpy'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)