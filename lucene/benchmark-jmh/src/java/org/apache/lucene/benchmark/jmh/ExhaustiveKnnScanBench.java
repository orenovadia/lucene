/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Benchmarks the exhaustive-scan branch of {@code Lucene99HnswVectorsReader.search} (the
 * "expected HNSW visits >= filtered count" fallback), which scores the accepted vectors in bulk
 * via {@code VectorScorer.Bulk}.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 2)
@Fork(1)
@State(Scope.Benchmark)
public class ExhaustiveKnnScanBench {
  static final int N = 200_000;
  static final int DIM = 384;
  static final String FIELD = "v";

  @Param({"dense", "sparse5"})
  public String density;

  @Param({"sel0.001", "sel0.05"})
  public String filter;

  IndexReader reader;
  org.apache.lucene.codecs.KnnVectorsReader knnReader;
  org.apache.lucene.index.FloatVectorValues vectorValues;
  AcceptDocs acceptDocs;
  float[] target;
  int k;

  Path tmpDir;
  Path prebuiltIndex;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    double selectivity = filter.equals("sel0.001") ? 0.001 : 0.05;
    k = selectivity == 0.001 ? 100 : 1024;
    boolean sparse = density.equals("sparse5");
    int numVectors = sparse ? N / 20 : N;

    // check.main mode / JMH runs can supply a prebuilt index root (per density) to avoid
    // rebuilding the index in every trial
    String indexRoot = System.getProperty("bench.indexRoot");
    Path prebuilt = prebuiltIndex != null ? prebuiltIndex
        : (indexRoot != null ? Path.of(indexRoot, "bench-idx-" + density) : null);
    if (prebuilt != null) {
      Files.createDirectories(prebuilt);
    }
    tmpDir = prebuilt != null ? prebuilt : Files.createTempDirectory("ExhaustiveKnnScanBench");
    Random rnd = new Random(42);
    MMapDirectory dir = new MMapDirectory(tmpDir);
    if (DirectoryReader.indexExists(dir) == false) {
      // Lucene104Codec's default KNN format is Lucene99HnswVectorsFormat, the patched reader.
      IndexWriterConfig cfg =
          new IndexWriterConfig()
              .setCodec(new org.apache.lucene.codecs.lucene104.Lucene104Codec())
              .setRAMBufferSizeMB(1024);
      try (IndexWriter writer = new IndexWriter(dir, cfg)) {
        for (int i = 0; i < N; i++) {
          if (sparse && i % 20 != 0) {
            continue;
          }
          float[] vec = new float[DIM];
          for (int d = 0; d < DIM; d++) {
            vec[d] = rnd.nextFloat() - 0.5f;
          }
          Document doc = new Document();
          doc.add(new KnnFloatVectorField(FIELD, vec, VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
    }

    reader = DirectoryReader.open(new MMapDirectory(tmpDir));
    LeafReaderContext leaf = reader.leaves().get(0);
    SegmentReader segReader = (SegmentReader) leaf.reader();
    if (reader.leaves().size() != 1) {
      throw new IllegalStateException("expected single segment");
    }
    knnReader = ((CodecReader) segReader).getVectorReader();
    vectorValues = segReader.getFloatVectorValues(FIELD);

    // Filter bits at the requested selectivity over maxDoc.
    int maxDoc = segReader.maxDoc();
    FixedBitSet bits = new FixedBitSet(maxDoc);
    Random frnd = new Random(7);
    int toAccept = (int) Math.round(selectivity * maxDoc);
    while (bits.cardinality() < toAccept) {
      bits.set(frnd.nextInt(maxDoc));
    }
    acceptDocs = AcceptDocs.fromLiveDocs(bits, maxDoc);

    target = new float[DIM];
    for (int d = 0; d < DIM; d++) {
      target[d] = rnd.nextFloat() - 0.5f;
    }

    // Assert the exhaustive branch (not HNSW) will run for these parameters.
    int graphSize = numVectors;
    int filteredDocCount = Math.min(acceptDocs.cost(), graphSize);
    int unfilteredVisit = HnswGraphSearcher.expectedVisitedNodes(k, graphSize);
    if (unfilteredVisit < filteredDocCount) {
      throw new IllegalStateException(
          "parameters would take the HNSW branch: k="
              + k
              + " expectedVisit="
              + unfilteredVisit
              + " filteredDocCount="
              + filteredDocCount);
    }
    System.out.println(
        "density="
            + density
            + " filter="
            + filter
            + " maxDoc="
            + maxDoc
            + " numVectors="
            + numVectors
            + " k="
            + k
            + " filteredDocCount="
            + filteredDocCount);
  }

  @TearDown(Level.Trial)
  public void tearDown() throws IOException {
    reader.close();
    if (prebuiltIndex == null) {
      org.apache.lucene.util.IOUtils.rm(tmpDir);
    }
  }

  @Benchmark
  public TopDocs searchExhaustive() throws IOException {
    TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE);
    knnReader.search(FIELD, target, collector, acceptDocs);
    return collector.topDocs();
  }

  /** Check mode: prints a digest of the collected top-K for cross-variant comparison. */
  public static void main(String[] args) throws Exception {
    ExhaustiveKnnScanBench bench = new ExhaustiveKnnScanBench();
    bench.density = System.getProperty("check.density", "dense");
    bench.filter = System.getProperty("check.filter", "sel0.001");
    String idx = System.getProperty("check.indexDir");
    if (idx != null) {
      bench.prebuiltIndex = Path.of(idx);
      Files.createDirectories(bench.prebuiltIndex);
    }
    bench.setup();
    TopDocs td = bench.searchExhaustive();
    StringBuilder sb =
        new StringBuilder()
            .append("digest ")
            .append(bench.density)
            .append('/')
            .append(bench.filter)
            .append(" totalHits=")
            .append(td.totalHits)
            .append(" hits:");
    for (var sd : td.scoreDocs) {
      sb.append(' ').append(sd.doc).append(':').append(sd.score);
    }
    System.out.println(sb);

    // Ground truth: brute-force scan of every *accepted* vector, top-k by cosine.
    record DS(int doc, float score) {}
    List<DS> all = new ArrayList<>();
    org.apache.lucene.util.Bits acceptedBits = bench.acceptDocs.bits();
    for (int ord = 0; ord < bench.vectorValues.size(); ord++) {
      int doc = bench.vectorValues.ordToDoc(ord);
      if (acceptedBits.get(doc) == false) {
        continue;
      }
      all.add(
          new DS(
              doc,
              org.apache.lucene.index.VectorSimilarityFunction.COSINE.compare(
                  bench.vectorValues.vectorValue(ord), bench.target)));
    }
    all.sort((a, b) -> Float.compare(b.score, a.score));
    int n = Math.min(bench.k, all.size());
    int docMismatches = 0;
    float maxScoreDelta = 0;
    for (int i = 0; i < n; i++) {
      DS gt = all.get(i);
      var sd = td.scoreDocs[i];
      if (gt.doc() != sd.doc) {
        docMismatches++;
        System.out.println("pos " + i + " doc mismatch: search=" + sd.doc + " brute=" + gt.doc()
            + " scores " + sd.score + " vs " + gt.score());
      }
      maxScoreDelta = Math.max(maxScoreDelta, Math.abs(gt.score() - sd.score));
    }
    System.out.println(
        "RESULT "
            + bench.density
            + '/'
            + bench.filter
            + " n="
            + n
            + " docMismatches="
            + docMismatches
            + " maxScoreDelta="
            + maxScoreDelta);
  }
}
