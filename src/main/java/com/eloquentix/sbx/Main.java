package com.eloquentix.sbx;

import org.deeplearning4j.models.featuredetectors.autoencoder.recursive.Tree;
import org.deeplearning4j.models.rntn.RNTN;
import org.deeplearning4j.models.rntn.RNTNEval;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.corpora.treeparser.TreeParser;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.activation.Tanh;

import java.io.File;
import java.util.List;

/**
 * Created by gabriel on 01.06.2015.
 */
public class Main {
    public static void main(String[] args) throws Exception{
        saveW2V();
        trainRNTN();
        evaluateRNTN();
    }

    private static void saveW2V() throws Exception {
        FolderSentenceIterator iterator =
                new FolderSentenceIterator("/home/gabriel/workspace/EL/py-classifier/corpus_ner_big/all/training");
        Word2Vec vec = new Word2Vec.Builder().
                layerSize(40).
                windowSize(10).
                minWordFrequency(1).
                iterate(iterator).
                iterations(3).
                learningRate(0.05).
                minLearningRate(0.001).
                tokenizerFactory(new DefaultTokenizerFactory()).build();
        vec.fit();
        SerializationUtils.saveObject(vec, new File("/tmp/vec.ser"));
    }

    private static void trainRNTN() throws Exception{
        Word2Vec vec = SerializationUtils.<Word2Vec>readObject(new File("/tmp/vec.ser"));
        RNTN rntn = new RNTN.Builder().
                setActivationFunction(new Tanh()).
                setUseTensors(false).
                setFeatureVectors(vec).
                build();
        FolderSentenceIterator iterator =
                new FolderSentenceIterator("/home/gabriel/workspace/EL/py-classifier/corpus_ner_big/all/training");
        TreeVectorizer treeVectorizer = new TreeVectorizer(new TreeParser());
        int i = 0;
        while (iterator.hasNext()){
            if(i++ % 100 == 0){
                System.out.println(String.format("Fitted %d sents", i));
            }
            List<Tree> treesWithLabels = treeVectorizer.getTreesWithLabels(iterator.nextSentence(),
                    iterator.currentLabel(),
                    iterator.currentLabels());
            rntn.fit(treesWithLabels);
        }
        SerializationUtils.saveObject(rntn, new File("/tmp/rntn.ser"));
    }

    private static void evaluateRNTN() throws Exception{
        RNTNEval eval = new RNTNEval();
        RNTN rntn = SerializationUtils.<RNTN>readObject(new File("/tmp/rntn.ser"));
        FolderSentenceIterator iterator =
                new FolderSentenceIterator("/home/gabriel/workspace/EL/py-classifier/corpus_ner_big/all/test");
        TreeVectorizer vectorizer = new TreeVectorizer(new TreeParser());
        int i = 0;
        while (iterator.hasNext()){
            if(i++ % 100 == 0){
                System.out.println(String.format("Evaluated %d sents", i));
            }
            List<Tree> treesWithLabels =
                    vectorizer.getTreesWithLabels(iterator.nextSentence(),
                            iterator.currentLabel(),
                            iterator.currentLabels());
            eval.eval(rntn, treesWithLabels);
        }
        System.out.println(eval.stats());
    }
}
