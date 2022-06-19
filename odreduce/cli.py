import argparse

import odreduce
from odreduce import pipeline
from odreduce import INFDIR, INPDIR, OUTDIR



def main():

#####################################################################
# Initiate parser
#

    parser = argparse.ArgumentParser(
                                     description="ODreduce: Observation and Data Reduction",
                                     prog='odreduce',
    )
    parser.add_argument('-version', '--version',
                        action='version',
                        version="%(prog)s {}".format(odreduce.__version__),
                        help="Print version number and exit."
    )


######################################################################
# Parent parser contains arguments and options common to all modes
#

    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('-c', '--cli', 
                               dest='cli',
                               help='This option should not be adjusted for current users',
                               default=True,
                               action='store_false',
    )
    parent_parser.add_argument('--in', '--input', '--inpdir', 
                               metavar='path',
                               dest='inpdir',
                               help='Input directory',
                               type=str,
                               default=INPDIR,
    )
    parent_parser.add_argument('--info', '--information',
                               metavar='path',
                               dest='info',
                               help='Path to star info',
                               type=str,
                               default=INFDIR,
    )
    parent_parser.add_argument('--out', '--outdir', '--output',
                               metavar='path',
                               dest='outdir',
                               help='Output directory',
                               type=str,
                               default=OUTDIR,
    )
    parent_parser.add_argument('-v', '--verbose', 
                               dest='verbose',
                               help='Turn on verbose output',
                               default=False, 
                               action='store_true',
    )


#####################################################################
# Initial and/or final data treatment + related processes
#

    main_parser = argparse.ArgumentParser(add_help=False)

    main_parser.add_argument('-d', '--show', '--display',
                             dest='show',
                             help='Show output figures',
                             default=False, 
                             action='store_true',
    )
    main_parser.add_argument('-s', '--save',
                             dest='save',
                             help='Do not save output figures and results.',
                             default=True, 
                             action='store_false',
    )
    main_parser.add_argument('--star', '--stars',
                             metavar='star',
                             dest='stars',
                             help='List of stars to process',
                             type=str,
                             nargs='*',
                             default=None,
    )
    main_parser.add_argument('-wave', '--wave_band',
                             metavar='wave',
                             dest='wave_band',
                             help='wave band of observation',
                             type=str,
                             nargs='*',
                             default=None,
    )
    main_parser.add_argument('-f', '--find_target', 
                               dest='find_target',
                               help='Turn on verbose output',
                               default=False, 
                               action='store_true',
    )


    sub_parser = parser.add_subparsers(title='ODreduce modes', dest='command')

#####################################################################
# pySYD mode loads in data for a single target but does not run
# -> still under development 
# idea is to have options to plot data, etc.
#

    parser_load = sub_parser.add_parser('load',
                                        conflict_handler='resolve',
                                        parents=[parent_parser], 
                                        formatter_class=argparse.MetavarTypeHelpFormatter,
                                        help='Load in data for a given target',  
                                        )

    parser_load.set_defaults(func=pipeline.main)



#####################################################################
# Run the main pySYD pipeline on 1 or more targets
#


    parser_run = sub_parser.add_parser('run',
                                       conflict_handler='resolve', 
                                       help='Run the main ODreduce pipeline',
                                       parents=[parent_parser, main_parser], 
                                       formatter_class=argparse.MetavarTypeHelpFormatter,
                                       )

    parser_run.set_defaults(func=pipeline.main)



    args = parser.parse_args()
    args = args.func(args)


if __name__ == '__main__':

    main()
