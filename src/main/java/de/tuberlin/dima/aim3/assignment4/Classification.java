/**
 * AIM3 - Scalable Data Mining -  course work
 * Copyright (C) 2014  Sebastian Schelter, Christoph Boden
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package de.tuberlin.dima.aim3.assignment4;


import com.google.common.collect.Maps;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.DataSource;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.FileSystem;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

public class Classification {
	
	public static final int k = 1; // For Laplace smoothing

  public static void main(String[] args) throws Exception {

    ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

    DataSource<String> conditionalInput = env.readTextFile(Config.pathToConditionals());
    DataSource<String> sumInput = env.readTextFile(Config.pathToSums());

    DataSet<Tuple3<String, String, Long>> conditionals = conditionalInput.map(new ConditionalReader());
    DataSet<Tuple2<String, Long>> sums = sumInput.map(new SumReader());

    // Test 
    DataSource<String> testData = env.readTextFile(Config.pathToTestSet());

    DataSet<Tuple3<String, String, Double>> classifiedDataPoints = testData.map(new Classifier())
        .withBroadcastSet(conditionals, "conditionals")
        .withBroadcastSet(sums, "sums");

    classifiedDataPoints.writeAsCsv(Config.pathToOutput(), "\n", "\t", FileSystem.WriteMode.OVERWRITE);
    
    // Secret
    DataSource<String> secretTestData = env.readTextFile(Config.pathToSecretTestSet());

    DataSet<Tuple3<String, String, Double>> secretDataPoints = secretTestData.map(new Classifier())
        .withBroadcastSet(conditionals, "conditionals")
        .withBroadcastSet(sums, "sums");

    secretDataPoints.writeAsCsv(Config.pathToSecretOutput(), "\n", "\t", FileSystem.WriteMode.OVERWRITE);

    env.execute();
  }

  public static class ConditionalReader implements MapFunction<String, Tuple3<String, String, Long>> {

    @Override
    public Tuple3<String, String, Long> map(String s) throws Exception {
      String[] elements = s.split("\t");
      return new Tuple3<String, String, Long>(elements[0], elements[1], Long.parseLong(elements[2]));
    }
  }

  public static class SumReader implements MapFunction<String, Tuple2<String, Long>> {

    @Override
    public Tuple2<String, Long> map(String s) throws Exception {
      String[] elements = s.split("\t");
      return new Tuple2<String, Long>(elements[0], Long.parseLong(elements[1]));
    }
  }


  public static class Classifier extends RichMapFunction<String, Tuple3<String, String, Double>>  {

     private final Map<String, Map<String, Long>> wordCounts = Maps.newHashMap();
     private final Map<String, Long> wordSums = Maps.newHashMap();

     @Override
     public void open(Configuration parameters) throws Exception {
    	 super.open(parameters);

    	 // IMPLEMENT ME
    	 // Take <Label, word, count> from broadcast and store in hashmap
    	 Collection<Tuple3<String, String, Long>> conditionals = getRuntimeContext().getBroadcastVariable("conditionals");
    	 for (Tuple3<String, String, Long> t : conditionals) {
    		 Map<String, Long> m = wordCounts.get(t.f0);
    		 if (m == null) {
    			 m = Maps.newHashMap();
    			 wordCounts.put(t.f0, m);
    		 }
    		 m.put(t.f1, t.f2);
    	 }

    	 // take <Label, total word count> from broadcast to hashmap
    	 Collection<Tuple2<String, Long>> sums = getRuntimeContext().getBroadcastVariable("sums");
    	 for (Tuple2<String, Long> t : sums) {
    		 wordSums.put(t.f0, t.f1);
    	 }
     }

     @Override
     public Tuple3<String, String, Double> map(String line) throws Exception {

       String[] tokens = line.split("\t");
       String label = tokens[0];
       String[] terms = tokens[1].split(",");

       double maxProbability = Double.NEGATIVE_INFINITY;
       String predictionLabel = "";

       for (String currentLabel : wordSums.keySet()) {
    	   double probability = 0;  
    	   for (String t : terms) {

    		   Long count_w_y = null;
    		   Map<String, Long> m = null;
    		   m = wordCounts.get(currentLabel);
    		   if (m != null) 
    			   count_w_y = m.get(t);
    		   if (count_w_y == null)
    			   	count_w_y = 0L;
    		   count_w_y += k; // smoothing
    		   
    		   Long count_w_not_y = wordSums.get(currentLabel);
    		   if (count_w_not_y == null)
    			   count_w_not_y = 0L;
    		   count_w_not_y += -count_w_y + (wordSums.size()-1)*k;
    		   
    		   probability += Math.log((double)count_w_y/(double)count_w_not_y);
    	   }
    	   if (probability > maxProbability) {
    		   maxProbability = probability;
    		   predictionLabel = currentLabel;
    	   }
       }

       return new Tuple3<String, String, Double>(label, predictionLabel, maxProbability);
     }
   }

}
