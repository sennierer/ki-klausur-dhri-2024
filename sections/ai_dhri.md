### AI in the text based DH
- Images to text (OCR + HTR)<!-- .element: class="fragment" -->
- Text to structured data (NLP)<!-- .element: class="fragment" -->
- Enrichment of Knowledgebases (GNNs, graph embeddings etc.)<!-- .element: class="fragment" -->
- Using LLMs:<!-- .element: class="fragment" -->
    - structured data to text<!-- .element: class="fragment" -->
    - Retrieval-Augmented Generation (RAG)<!-- .element: class="fragment" -->

---

### NLP 
- some experience with spaCy and Flair<!-- .element: class="fragment" -->
- (re)trained NER models on ÖBL/APIS data (F1 ~85%)<!-- .element: class="fragment" -->
- model for expanding abbreviations<!-- .element: class="fragment" --> 
- model/pipeline for finding the semantic relation of depicted person and NE (F1 ~65%)<!-- .element: class="fragment" -->
- no ML based solution for named entity linking<!-- .element: class="fragment" -->

---

### LLM examples
- Used the GPU at jupyter.arz.oeaw.ac.at<!-- .element: class="fragment" -->
- Using OLLAMA:<!-- .element: class="fragment" -->
    - open-source server for running LLMs<!-- .element: class="fragment" -->
    - provides terminal based prompt<!-- .element: class="fragment" -->
    - plus RestAPI<!-- .element: class="fragment" -->
- Examples created LLAMA3:70b<!-- .element: class="fragment" -->

---

### Example I: Bio text from MINE data for Ludwig Boltzmann

        You are a historian working on the history of the Austrian Academy of Sciences. 
        Please write a biography on Ludwig Boltzmann using only this data: 
        ... Ludwig
        ... Boltzmann
        ...
        ... Physiker, Philosoph
        ... Geboren 20.02.1844 (Wien)
        ... Gestorben 05.09.1906
        ...
        ...     o. Univ.-Prof. für mathematische Physik: Karl-Franzens-Universität GrazR
        ...     o. Univ.-Prof. für Mathematik: Universität WienR
        ...     Vorstand/Vorständin: Physikalisches Institut (Graz)R
        ...
        ...     1872 zur Wahl zum kM I der math.-nat. Klasse vorgeschlagen von: Josef Stefan, Anton Winckler, Eduard Fenzl, Karl Littrow, August Ema
        ... nuel Reuss, Victor Lang, Ernst Brücke, Anton Schrötter-Kristelli
        ...     nicht gewählt
        ...     1874 zur Wahl zum kM I der math.-nat. Klasse vorgeschlagen von: Josef Stefan, Victor Lang, Josef Loschmidt, Heinrich Hlasiwetz, Erns
        ... t Brücke, Anton Winckler, Karl Littrow, Anton Schrötter-Kristelli
        ...     als kM I gewählt und bestätigt am 9.7.1874
        ...     1879 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Josef Stefan, Ernst Brücke, Josef Loschmidt, Edmund Weiss, Victor Lang
        ...
        ...     nicht gewählt
        ...     1880 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Josef Stefan, Julius Ferdinand Hann, Ernst Brücke, Victor Lang, Edmund
        ...  Weiss, Josef Loschmidt
        ...     nicht gewählt
        ...     1882 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Josef Stefan, Edmund Weiss, Victor Lang, Julius Ferdinand Hann
        ...     nicht gewählt
        ...     1885 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Adolf Lieben, Edmund Weiss, Ernst Mach, Josef Stefan, Josef Loschmidt,
        ...  Victor Lang
        ...     als wM gewählt und ernannt am 14.7.1885
        ...     1891 umgewidmet zum kM A der math.-nat. Klasse
        ...     1895 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Victor Lang, Franz Karl Mertens, Gustav Escherich, Gustav Tschermak-Se
        ... ysenegg, Julius Ferdinand Hann, Carl Claus
        ...     als wM gewählt und ernannt am 7.8.1895
        ...     1900 umgewidmet zum kM A der math.-nat. Klasse
        ...     1902 umgewidmet zum kM I der math.-nat. Klasse
        ...     1904 zur Wahl zum wM der math.-nat. Klasse vorgeschlagen von: Victor Lang, Zdenko Hans Skraup, Franz Karl Mertens, Edmund Weiss, Vic
        ... tor Gilbert Ebner-Rofenstein, Julius Ferdinand Hann, Franz Steindachner, Gustav Escherich, Adolf Lieben
        ...     als wM gewählt und ernannt am 10.9.1904
        ...
        ...     zur Wahl vorgeschlagen:
        ...         Klemensiewicz, Rudolf (1890)
        ...         Exner, Franz Serafin (1896)
        ...         Zuckerkandl, Emil (1896)
        ...         Ludwig, Ernst (1896)
        ...         Haberlandt, Gottlieb J. (1896)
        ...         Eder, Josef Maria (1896)
        ...         Hepperger, Josef (1896)
        ...         Exner, Karl (1896)
        ...         Exner, Karl (1897)
        ...         Becke, Friedrich (1898)
        ...         Lecher, Ernst (1898)
        ...         Stolz, Otto (1899)
        ...         Ludwig, Ernst (1899)
        ...         Wirtinger, Wilhelm August (1905)
        ...         Jaumann, Gustav (1905)
        ...         Herzig, Josef (1905)
        ...
        ...     Mitglied der folgenden Kommissionen/Kuratorien:
        ...         Kommission zur Herausgabe der mathematischen Enzyklopädie (1895-1900, 1904-1906)
        ...         Erdbebenkommission (1898-1900)
        ...         Kommission für die Gründung eines phonographischen Archives (1899-1906)
        ...         Kommission für Sonnenforschung (1905-1906)
        ...
        ...     Ausgezeichnet mit folgenden Akademiepreisen:
        ...         Freiherr-von-Baumgartner'scher-Preis 1875
        ...
        ...         "Der zweite Hauptsatz der mechanischen Wärmetheorie" (Feierliche Sitzung des Jahres 1886, 28.05.1886)

---

### Bio produced by LLAMA3:70b

> Ludwig Boltzmann was born on February 20, 1844, in Vienna, Austria-Hungary. He would go on to become one of the most influential
physicists and philosophers of his time.
Boltzmann's academic career began at the University of Graz, where he held the position of ordinary university professor for
mathematical physics from 1873. In 1876, he moved to the University of Vienna, becoming an ordinary university professor for
mathematics.
Despite being a prominent figure in the scientific community, Boltzmann's path to membership in the Austrian Academy of Sciences was
not without its obstacles. He was first proposed for membership in 1872 by a group of esteemed scientists, including Josef Stefan and
Anton Winckler. However, he was not elected at that time. It wasn't until two years later, in 1874, that Boltzmann was successfully
elected as a corresponding member (kM I) of the mathematical-natural sciences class.
Over the next several decades, Boltzmann would continue to play an active role in the Academy, serving on various commissions and
committees. He was proposed for election as a full member (wM) multiple times, finally achieving this distinction in 1885. He would
later be re-elected as a corresponding member in 1891 and again as a full member in 1895.
....<!-- .element: style="font-size:0.8em" -->

---

### Example II: RAG using ÖBL Bios (nomic-embed-text / LLAMA3:70b)

- Vectorstore from all ÖBL Bios (XMLs)<!-- .element: class="fragment" -->
- Embeddings created using nomic-embed-text with OLLAMA<!-- .element: class="fragment" -->
- Chunk size of 500 chars with 50 overlapp (needs probably bigger sizes)<!-- .element: class="fragment" -->
- No usage of structured data<!-- .element: class="fragment" -->
- "blind" extraction of text from XMLs<!-- .element: class="fragment" -->
- LLAMA3:70b got 50 snippets as context for answering the question<!-- .element: class="fragment" -->

---

### Example

    Für was war das Künstlerhaus berühmt?
    According to the provided context, the Künstlerhaus (Artists House) in Vienna 
    was a renowned institution where many artists were members and exhibited their 
    works. Several artists mentioned in the text were members of the Künstlerhaus, 
    including S., H., T., and others.

    The Künstlerhaus was likely famous for being a hub for artistic activity, 
    exhibitions, and community in Vienna. It provided a platform for artists to 
    showcase their work, exchange ideas, and collaborate with each other. The institution's 
    significance is reflected in the fact that many notable artists were associated with it, 
    and it continues to be mentioned throughout the text as an important part of their careers.

    In summary, the Künstlerhaus was berühmt (famous) for being a prominent artistic institution 
    in Vienna, fostering creativity, innovation, and community among its members.

    oebl-harmonized-v2/Fraknoi_Wilhelm_1843_1924.xml:
    und ein Künstlerhaus für ung. Gelehrte und Künstler.

    oebl-harmonized-v2/Six_Michael_1874_1938.xml:
    Wiens (Künstlerhaus), wo er auch regelmäßig ausstellte.

    oebl-harmonized-v2/Mielichhofer_Alphons_1863_1929.xml:
    Kunstver. (1894) und im Wr. Künstlerhaus, dessen Mitgl. er ab 1894 war.

    oebl-harmonized-v2/Onken_Karl-Eduard_1846_1934.xml:
    der Ges. bildender Künstler Wiens Künstlerhaus. Seine Gemälde befinden sich hauptsächlich in Privatbesitz, aber auch in Galerien in Oldenburg, Prag und Wien.


---

### Things to work on/discuss (from my perspective)
- Own hardware (GPUs) vs using hosted solutions<!-- .element: class="fragment" -->
    - if own hardware, how to facilitate the usage of the GPUs?<!-- .element: class="fragment" -->
    - if hosted solutions, how to finance and manage access?<!-- .element: class="fragment" -->
- Use of AI in process of creating data only vs also in running applications<!-- .element: class="fragment" -->
- Readymade solutions/workflows for standard tasks (e.g. create embeddings with OLLAMA)<!-- .element: class="fragment" -->

