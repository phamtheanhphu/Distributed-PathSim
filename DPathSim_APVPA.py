import os, timeit
import networkx as nx
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from time import gmtime, strftime

class DPathSim_APVPA():

    def __init__(self, dblp_graph, dblp_graphframe, source_author_node_id, output_file_path):

        self.dblp_graph = dblp_graph
        self.dblp_graphframe = dblp_graphframe
        self.source_author_node_id = source_author_node_id

        self.author_sim_scores = {}
        self.author_id_name_maps = {}

        for p, d in self.dblp_graph.nodes(data=True):
            if d['node_type'] == 'author':
                if p != self.source_author_node_id:
                    self.author_sim_scores.update({p: 1})
                self.author_id_name_maps.update({p: d['label']})

        # logging
        self.output_file = open(output_file_path, 'a', encoding='utf-8')
        self.overall_start_time = timeit.default_timer()

    def run(self):

        source_author_global_walk = self.metapath_global_walk(self.source_author_node_id)

        print('Source author global walk: {}'.format(source_author_global_walk))
        self.output_file.write('Source author global walk: {}\n'
                               .format(source_author_global_walk))

        for target_author_node_id in self.author_sim_scores.keys():
            start_time = timeit.default_timer()

            pairwise_node_walk = self.metapath_pairwise_walk(
                self.source_author_node_id, target_author_node_id)

            print('Pairwise authors walk {}: {}'.format(target_author_node_id, pairwise_node_walk))
            self.output_file.write('Pairwise authors walk {}: {}\n'
                                   .format(target_author_node_id, pairwise_node_walk))

            target_author_global_walk = self.metapath_global_walk(target_author_node_id)
            print('Target author global walk: {}'.format(target_author_global_walk))
            self.output_file.write('Target author global walk: {}\n'
                                   .format(target_author_global_walk))

            sim_score = 2 * (pairwise_node_walk) / \
                        (source_author_global_walk + target_author_global_walk)

            self.author_sim_scores.update({target_author_node_id: sim_score})

            print('Sim score {} - {}: {}'.format(self.author_id_name_maps[source_author_node_id],
                                                 self.author_id_name_maps[target_author_node_id],
                                                 sim_score))
            self.output_file.write('Sim score {} - {}: {}\n'.format(self.author_id_name_maps[source_author_node_id],
                                                                    self.author_id_name_maps[target_author_node_id],
                                                                    sim_score))

            self.output_file.write('***Stage done in: {}\n'.format(timeit.default_timer() - start_time))
            self.output_file.write('---\n')
            self.output_file.flush()

        self.output_file.write('***Overall done in: {}\n'.format(timeit.default_timer() - self.overall_start_time))
        self.output_file.close()

    def metapath_global_walk(self, start):

        motifs = self.dblp_graphframe.find(
            "(author_1)-[e1]->(paper_1); "
            "(paper_1)-[e2]->(venue); "
            "(paper_2)-[e3]->(venue); "
            "(author_2)-[e4]->(paper_2)") \
            .filter("author_1.id = '{}'".format(start)) \
            .filter("paper_1.node_type = 'paper'") \
            .filter("paper_2.node_type = 'paper'") \
            .filter("venue.node_type = 'venue'") \
            .filter("e1.relationship = 'author_of'") \
            .filter("e2.relationship = 'submit_at'") \
            .filter("e3.relationship = 'submit_at'") \
            .filter("e4.relationship = 'author_of'")

        total_path = motifs.select('*').distinct().count()

        return int(total_path)

    def metapath_pairwise_walk(self, source, target):

        motifs = self.dblp_graphframe.find(
            "(author_1)-[e1]->(paper_1); "
            "(paper_1)-[e2]->(venue); "
            "(paper_2)-[e3]->(venue); "
            "(author_2)-[e4]->(paper_2)") \
            .filter("author_1.id = '{}'".format(source)) \
            .filter("author_2.id = '{}'".format(target)) \
            .filter("paper_1.node_type = 'paper'") \
            .filter("paper_2.node_type = 'paper'") \
            .filter("venue.node_type = 'venue'") \
            .filter("e1.relationship = 'author_of'") \
            .filter("e2.relationship = 'submit_at'") \
            .filter("e3.relationship = 'submit_at'") \
            .filter("e4.relationship = 'author_of'")

        total_path = motifs.select('*').distinct().count()

        return int(total_path)


if __name__ == '__main__':

    def read_dblp_nx_file(dblp_graph_file_path):

        dblp_graph = nx.read_gexf(dblp_graph_file_path)
        vertices = []
        edges = []

        for p, d in dblp_graph.nodes(data=True):
            vertices.append((p, d['label'], d['node_type']))

        for s, t, d in dblp_graph.edges(data=True):
            edges.append((s, t, d['label']))

        print('Total nodes: {}'.format(len(vertices)))
        print('Total edges: {}'.format(len(edges)))

        return dblp_graph, vertices, edges


    def find_author_node_id_by_name(dblp_graph, author_name):
        for p, d in dblp_graph.nodes(data=True):
            if 'label' in d:
                if d['label'] == author_name:
                    return p
        return None


    # DBLP data local file in .gexf format
    # dblp_graph_file_path = 'dblp/dblp_large.gexf'
    dblp_graph_file_path = 'dblp/dblp_small.gexf'
    dblp_graph, vertices, edges = read_dblp_nx_file(dblp_graph_file_path)

    # declare the graphframe libraries
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark-shell"
    )

    # create the spark session
    spark_session = SparkSession \
        .builder \
        .appName("Spark_DPathSim") \
        .getOrCreate()

    # create spark context
    sqlContext = SQLContext(spark_session)

    # generate sql-context for vertices & edges
    gf_vertices = sqlContext \
        .createDataFrame(vertices, ["id", "label", "node_type"])
    gf_edges = sqlContext \
        .createDataFrame(edges, ["src", "dst", "relationship"])

    # mapping spark sql-context -> graphframes
    from graphframes import *

    dblp_graphframe = GraphFrame(gf_vertices, gf_edges)

    # source author for computing similarity
    source_author_name = 'Jiawei Han'
    source_author_node_id = find_author_node_id_by_name(dblp_graph, source_author_name)

    # output log file
    dblp_graph_file_path = 'output/d_pathsim_output_{}.log' \
        .format(strftime("%Y%m%d_%H%M%S", gmtime()))

    # run the distributed PathSim
    d_pathsim = DPathSim_APVPA(dblp_graph, dblp_graphframe, source_author_node_id, dblp_graph_file_path)
    d_pathsim.run()
