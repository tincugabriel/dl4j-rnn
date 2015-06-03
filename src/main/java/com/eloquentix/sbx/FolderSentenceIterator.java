package com.eloquentix.sbx;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by gabriel on 01.06.2015.
 */
public class FolderSentenceIterator implements LabelAwareSentenceIterator {
    public List<File> files;
    public int fileIndex = 0;
    public int sentenceIndex = 0;
    public String[] currentSentences;
    public List<String> currentLabels;
    public String currentLabel;
    public FolderSentenceIterator(String folderName) throws Exception{
        files = Arrays.asList(new File(folderName).listFiles(new FileFilter() {
            @Override
            public boolean accept(File file) {
                return file.isFile() && file.getName().endsWith("txt");
            }
        }));
    }

    public int getSentenceIndex() {
        return sentenceIndex;
    }

    public void setSentenceIndex(int sentenceIndex) {
        this.sentenceIndex = sentenceIndex;
    }

    public synchronized String[] getCurrentSentences() {
        return currentSentences;
    }

    public synchronized void setCurrentSentences(String[] currentSentences) {
        this.currentSentences = currentSentences;
    }

    public synchronized List<String> getCurrentLabels() {
        return currentLabels;
    }

    public synchronized void setCurrentLabels(List<String> currentLabels) {
        this.currentLabels = currentLabels;
    }

    public synchronized String getCurrentLabel() {
        return currentLabel;
    }

    public synchronized void setCurrentLabel(String currentLabel) {
        this.currentLabel = currentLabel;
    }

    @Override
    public String currentLabel() {
        return getCurrentLabel();
    }

    @Override
    public List<String> currentLabels() {
        return getCurrentLabels();
    }

    @Override
    public String nextSentence() {
        currentLabel = null;
        if (getCurrentSentences() == null){
            getNextSentenceBatch();
        }
        if(sentenceIndex == getCurrentSentences().length){
            fileIndex++;
            getNextSentenceBatch();
        }
        String sentence = getCurrentSentences()[sentenceIndex];
        sentenceIndex++;
        String[] triples = sentence.split("\n");
        StringBuilder result = new StringBuilder();
        int index = 0;
        List<String> labels = new ArrayList<String>();
        for(String triple : triples){
            String[] split = triple.split(" ");
            if(split.length < 3){
                continue;
            }
            String word = split[0];
            String tag = split[2];
            if(tag.toLowerCase().endsWith("prod")){
                labels.add("PROD");
                setCurrentLabel("PROD");
            } else {
                labels.add(tag);
            }
            result.append(word).append(' ');
        }
        setCurrentLabels(labels);
        if(null == getCurrentLabel()){
            setCurrentLabel("O");
        }
        String s = result.toString();
        s = s.trim();
        if(s.length() == 0){
            return nextSentence();
        }
        return s;
    }

    private void getNextSentenceBatch() {
        try {
            setCurrentSentences(new String(Files.readAllBytes(files.get(fileIndex).toPath())).split("\n\n"));
            sentenceIndex = 0;
        } catch (IOException e){
            throw new RuntimeException();
        }
    }

    @Override
    public boolean hasNext() {
        return fileIndex < files.size() -1 || sentenceIndex < getCurrentSentences().length - 1;
    }

    @Override
    public void reset() {
        sentenceIndex = 0;
        fileIndex = 0;
        setCurrentSentences(null);
    }

    @Override
    public void finish() {

    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor sentencePreProcessor) {

    }

    public static void main(String[] args) throws Exception{
        FolderSentenceIterator iterator = new FolderSentenceIterator("/home/gabriel/workspace/EL/py-classifier/corpus_ner");
        Word2Vec vec = new Word2Vec.Builder().iterate(iterator).batchSize(1000).iterations(10).layerSize(30).windowSize(7).learningRate(0.025)
                .minLearningRate(0.001).build();
        vec.fit();
    }
}
