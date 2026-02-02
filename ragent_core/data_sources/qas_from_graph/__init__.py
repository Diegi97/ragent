import traceback

if __name__ == '__main__':
    import logging

    from ragent_core.data_sources.qas_from_graph.graph import GraphStore, GraphExpander, GraphVisualizer
    from ragent_core.data_sources.qas_from_graph.qa_builder import QABuilder
    from ragent_core.data_sources.qas_from_graph.entity import EntityBuilder
    from ragent_core.data_sources.qas_from_graph.sparql_generator import SparqlBuilder
    from ragent_core.data_sources.qas_from_graph.expander_policies import GraphSingleResultPolicyExpander

    logging.basicConfig(level=logging.INFO)

    graph_store = GraphStore()
    expander = GraphExpander(graph_store)
    qa_builder = QABuilder(graph_store)
    spark_builder = SparqlBuilder(graph_store)
    visualizer = GraphVisualizer(graph_store)

    expander.init_graph()


    policy_expander = GraphSingleResultPolicyExpander(expander, spark_builder)
    answer_node, answer_attrs = graph_store.get_random_node()
    try:
        success = policy_expander.width_expand_until_single_result(answer_node, max_rounds=50)
    except Exception as e:
        print(traceback.format_exc())

    # expander.expand(times=3)
    # expander.expand(times=3)
    visualizer.draw()

    question = qa_builder.build_from_node(answer_node)
    sparql = spark_builder.build_from_node(answer_node)


    print('QUESTION:\n', question)
    print('ANSWER NODE:\n', answer_node, "ANSWER ATTRS:\n", answer_attrs)
    print('SPARQL:\n', sparql)
