# PFPython
Notebook para cinemática e dinâmica simbólica de robôs usando SymPy (fluxo 100% simbólico, sem backend numérico). Agora suporta
vetor de eixos (x/y/z) para definir a direção de rotação ou translação de cada junta.

- Valide o vetor de eixos com `validate_axes`, por exemplo `['z','y','y','z','y','z']` para 6 DOF, e alinhe-o com a lista de
  juntas antes de calcular jacobianos e dinâmicas.
