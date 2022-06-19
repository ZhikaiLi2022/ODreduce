# Import relevant, local ODreduce modules
import odreduce
from odreduce import utils
from odreduce import plots
from odreduce.target import Target


#####################################################################
# Main function that assigns functions for different modes 
#

def main(args=None):
   # Load in relevant information and data
   args = load(args)
   if args.command == 'run':
       run(args)
   else:
       pass

def load(args, star=None, verbose=False, command='run'):
    # Load relevant ODreduce parameters
    args = utils.get_info(args)
    return args

def run(args):
    # Run single batch of stars
    Target(args.params['stars'], args)
