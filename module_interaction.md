# Interakcje międzymodułowe

Każdy moduł powinien w sposób łatwy współpracować z każdym innym modułem. W tym celu powinniśmy mieć pewne API w każdym module tak, aby po jego napisaniu inne modułu (inne osoby) w sposób łatwy mogły z niego korzystać bez potrzeby dokładnej znajomości tego co jest w środku. Oczywiście super będzie jeśli każdy będzie wiedział co dokładnie każdy inny moduł robi, ale nie powinno to być wymagane do samej pracy nad własnym modułem.

Poniżej jest lista z założeniami jakie każdy moduł powinien spełniać.

**Moduł danych**

- `get_dataset(size, ratio_labeled, ratio_positive)`
    - zwraca losowy podzbiór danych z PCAM określony przez podane parametry
    - size: rozmiar datasetu
    - ratio_labeled: stosunek oznaczonych danych do wszystkich danych
    - ratio_positive: stosunek danych z pozytywnej klasy (rak) do wszystkich danych
    - (do dyskusji) dobrze jeśli oba zbiory, labeled i unlabeled, będą miały taki sam stosunek klas pozytywnych

**Moduł modelu**

**Moduł zapytań**

**Moduł wyroczni**

**Moduł oceny**