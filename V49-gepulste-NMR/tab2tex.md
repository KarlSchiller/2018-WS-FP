# Tab2Tex
Tab2Tex ist für die Automatisierung der Erstellung von Tabellen mit python, welche direkt
mit LuaLaTeX übersetzt werden können.

## Einbinden
Die Funktion `make_table` kann in einer anderen python Datei mit
`from tab2tex import make_table`
eingebunden werden.

## Parameter

    * header: Im header wird eine Liste mit den Namen der einzelnen Spalten übergeben.
      Diese Namen müssen also strings sein.
      Soll ein Name in der Matheumgebung sein,
      so sollte das erste und letzte Zeichen des Strings jeweils ein $ sein.
      Des weiteren kann durch eine Einheit geteilt werden,
      indem im String ein Leerzeichen, / und noch ein Leerzeichen stehen.
      Der Ausdruck dahinter wird in in eine si-Umgebung des Pakets `siunitx` geschrieben.
    * places: Hier ist eine Liste mit den anzugebenden Stellen ein zu geben. Sollten die Einträge
      der Spalte strings und keine Zahlen sein, so wird der jeweilige Eintrag ignoriert.
      Bei uncertainty.unumpy.uarray-Einträgen in data wird hier ein Tupel mit den
      zugehörigen Stellen angegeben (das Tupel hat also zwei Einträge).
      Die Einträge sollten entweder float oder None (für string-Einträge) sein.
      Eingegebene Daten werden automatisch auf die angegebenen Stellen gerundet
    * data: Hier werden die eigentlichen daten in einer Liste übergeben. Jeder Eintrag steht
      für eine Spalte. Es können auch uncertainty.unumpy.uarray übergeben werden.
      Bei unterschiedlichen langen Einträgen wird eine Warnung ausgegeben.
      Es dürfen keine Leeren Einträge oder eine komplett leeres data-array übergeben werden.
    * caption: Die Überschrift/Unterschrift der Tabelle. Wird dem `\caption{}` Befehl von
      LuaLaTeX übergeben. Muss ein String sein.
    * label: Das LuaLaTeX-Label der Tabelle. Muss ein String sein
    * filename: Der Speicherort der erstellten Tabelle. Muss ein String sein.