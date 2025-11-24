# PFPython
Notebook para cinemática e dinâmica simbólica de robôs usando SymPy (fluxo 100% simbólico, sem backend numérico). Agora suporta
vetor de eixos (x/y/z) para definir a direção de rotação ou translação de cada junta e inclui um exemplo completo para um UVMS
de 12 DOF (ROV + braço) com matriz de excentricidades configurável.

- Valide o vetor de eixos com `validate_axes`, por exemplo `['z','y','y','z','y','z']` para 6 DOF, e alinhe-o com a lista de
  juntas antes de calcular jacobianos e dinâmicas.
- A função `dynamics(..., parallel=True, processes=4)` paraleliza o cálculo simbólico de energias e equações de Lagrange em
  múltiplos processos de CPU, mantendo apenas operações simbólicas (útil para robôs com muitos elos). Ajuste `processes` para
  controlar o número de núcleos usados.
- A cinemática direta usa apenas os parâmetros DH (sem deslocar por excentricidades); a `matriz_excentricidades` move o centro
  de massa de **cada** elo apenas na etapa dinâmica (M/C/G/τ).
- Use `parse_axis_order` para aceitar ordens mistas de juntas prismáticas (Dx/Dy/Dz) e rotacionais (x/y/z); combine com
  `matriz_excentricidades` para gerar automaticamente os elos de um UVMS e calcular M/C/G/τ de forma paralelizada.
- Ative mensagens de debug com `debug=True` em `forward_kinematics`/`spatial_jacobians`/`dynamics` para imprimir o término do
  cálculo de cada elo (cinemática e dinâmica) com `flush` imediato, inclusive durante execução paralela no Colab, útil para
  acompanhar a progressão simbólica.
