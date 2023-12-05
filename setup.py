from distutils.core import setup
setup(
  name = 'climb_interp',
  packages = ['climb_interp'],
  version = '0.1',
  license='MIT',
  description = 'Complex explosion and flammable data interpolation!',
  author = 'Jonathan Motta',
  author_email = 'jonathangmotta98@gmail.com',
  url = 'https://github.com/Safe-Solutions-Engenharia/climb_interp.git',
  download_url = 'https://github.com/Safe-Solutions-Engenharia/climb_interp/archive/refs/tags/v_0_1.tar.gz',
  keywords = ['interpolation', 'explosion', 'flammable', 'data', 'linear', 'exponential', 'logarithmic'],
  install_requires=['scipy',
                    'matplotlib',
                    'cvxpy',
                    'warnings',
                    'logging'
                    ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)