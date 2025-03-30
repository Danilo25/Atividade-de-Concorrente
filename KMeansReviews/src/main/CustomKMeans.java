package main;

import java.io.*;
import java.util.*;

public class CustomKMeans {

    public static void main(String[] args) throws IOException {
        String filePath = "D:/DataSets/usar/train.csv";
        List<String> reviews = loadData(filePath);

        if (reviews.isEmpty()) {
            System.out.println("Nenhuma avaliação encontrada.");
            return;
        }

        // Converter textos para representações numéricas (número de palavras)
        int[] featureVectors = textToWordCount(reviews);

        // Aplicar K-Means com 3 clusters
        int k = 3;
        int[] clusters = kMeans(featureVectors, k);

        // Agrupar reviews por cluster e calcular os centróides finais
        Map<Integer, List<String>> clusteredReviews = new HashMap<>();
        Map<Integer, List<Integer>> clusteredWordCounts = new HashMap<>();
        int[] centroids = new int[k];

        for (int i = 0; i < reviews.size(); i++) {
            clusteredReviews.putIfAbsent(clusters[i], new ArrayList<>());
            clusteredWordCounts.putIfAbsent(clusters[i], new ArrayList<>());
            clusteredReviews.get(clusters[i]).add(reviews.get(i));
            clusteredWordCounts.get(clusters[i]).add(featureVectors[i]);
        }

        // Calcular centróides finais
        for (int cluster : clusteredWordCounts.keySet()) {
            List<Integer> wordCounts = clusteredWordCounts.get(cluster);
            centroids[cluster] = wordCounts.stream().mapToInt(Integer::intValue).sum() / wordCounts.size();
        }

        // Mostrar os três primeiros de cada cluster com suas contagens de palavras
        for (int cluster : clusteredReviews.keySet()) {
            System.out.println("Cluster " + cluster + " (Centróide: " + centroids[cluster] + " palavras):");
            List<String> reviewsInCluster = clusteredReviews.get(cluster);
            List<Integer> wordCounts = clusteredWordCounts.get(cluster);
            for (int i = 0; i < Math.min(3, reviewsInCluster.size()); i++) {
                System.out.println("- [" + wordCounts.get(i) + " palavras] " + reviewsInCluster.get(i));
            }
            System.out.println("------------------------");
        }
    }

    public static List<String> loadData(String filePath) throws IOException {
        List<String> reviews = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",", 3);
            if (parts.length > 2) {
                reviews.add(parts[2].trim());
            }
        }
        br.close();
        return reviews;
    }

    public static int[] textToWordCount(List<String> texts) {
        int[] wordCounts = new int[texts.size()];
        for (int i = 0; i < texts.size(); i++) {
            wordCounts[i] = texts.get(i).split(" ").length;
        }
        return wordCounts;
    }

    public static int[] kMeans(int[] data, int k) {
        int n = data.length;
        int[] centroids = new int[k];
        int[] labels = new int[n];
        Random rand = new Random();

        for (int i = 0; i < k; i++) {
            centroids[i] = data[rand.nextInt(n)];
        }

        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < n; i++) {
                int closest = 0;
                int minDist = Math.abs(data[i] - centroids[0]);
                for (int j = 1; j < k; j++) {
                    int dist = Math.abs(data[i] - centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        closest = j;
                    }
                }
                if (labels[i] != closest) {
                    labels[i] = closest;
                    changed = true;
                }
            }

            int[] sum = new int[k];
            int[] count = new int[k];
            for (int i = 0; i < n; i++) {
                sum[labels[i]] += data[i];
                count[labels[i]]++;
            }
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    centroids[j] = sum[j] / count[j];
                }
            }
        } while (changed);

        return labels;
    }
}
