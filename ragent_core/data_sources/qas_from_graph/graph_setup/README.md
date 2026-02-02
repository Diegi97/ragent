### Steps to set up a graph database using Virtuoso
1. Download  DBPedia data

   1. Ontology

      http://mappings.dbpedia.org/server/ontology/dbpedia.owl

   2. Data

         1. Fetch the query

         `query=$(curl -H "Accept:text/sparql" https://databus.dbpedia.org/dbpedia/collections/latest-core)`

         2. Download the files
   
         `files=$(curl -X POST -H "Accept: text/csv" --data-urlencode "query=${query}" https://databus.dbpedia.org/sparql | tail -n +2 | sed 's/\r$//' | sed 's/"//g')`


2. Run Virtuoso docker container (change the volumes path for your directory where you downloaded the data)
       `./run_virtuoso.sh`

3. Load the data into Virtuoso.

    From the client interface (`isql-vt`) run the following commands to load the data

   1. Ontology 

           ld_dir('/opt/virtuoso-opensource/database/init_onto', '*.owl', 'http://dbpedia.org');
           rdf_loader_run();
           checkpoint;

   2. Data

           ld_dir('/opt/virtuoso-opensource/database/init_data', '*.bz2', 'http://dbpedia.org');
           rdf_loader_run();
           checkpoint;

Notes:
- Currently, the directory is hardcoded in the script: `/d/linux_second/virtuoso_db` (you can change it in the `run_virtuoso.sh` script)