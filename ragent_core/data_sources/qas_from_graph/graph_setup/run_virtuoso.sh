docker run \
    --name my_virtdb \
    --interactive \
    --tty \
    --env DBA_PASSWORD=mysecret \
    --env VIRT_SPARQL_MAXQUERYCOSTESTIMATIONTIME=3600 \
    --env VIRT_SPARQL_MAXQUERYEXECUTIONTIME=360 \
    --env VIRT_PARAMETERS_THREADTHRESHOLD=20 \
    --env VIRT_PARAMETERS_THREADSPERQUERY=5 \
    --env VIRT_PARAMETERS_ASYNCQUEUEMAXTHREADS=20 \
    --publish 1111:1111 \
    --publish  8890:8890 \
    --volume /d/linux_second/virtuoso_db:/database \
    --volume /d/linux_second/dbpediaowl:/opt/virtuoso-opensource/database/init_onto \
    --volume /d/linux_second/dbpedia:/opt/virtuoso-opensource/database/init_data \
    openlink/virtuoso-opensource-7:latest
