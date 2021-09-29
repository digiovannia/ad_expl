# Code for "Balancing Adapatability and Non-exploitability in Repeated Games"

Workflow:
1) Run "python launcher_erg_equiv.py $i" for i in {0..35199}, parallelizing as needed
2) Create a subdirectory called "summaries"
3) When in the same directory as the "summaries" subdirectory, run "python results_laff.py". You will be prompted to enter the names of the directories containing the experiment results (within the subdirectory "barg_results_pre"); by default this will just be the directory in which you ran step 1.
