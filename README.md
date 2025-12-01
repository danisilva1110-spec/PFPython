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
- O helper `build_links_from_matrices` replica a interface matricial do projeto MATLAB: passe matrizes de DH,
  excentricidades, tensores de inércia (simetrizados automaticamente) e as ordens de rotação/translation de cada junta para
  obter os objetos `LinkParameters`. Em seguida, `equations_of_motion_from_matrices` entrega M/C/H/G já prontos a partir
  dessas matrizes, mantendo 100% das operações simbólicas.
- Ative mensagens de debug com `debug=True` em `forward_kinematics`/`spatial_jacobians`/`dynamics` para imprimir o término do
  cálculo de cada elo (cinemática e dinâmica) com `sys.stdout.write` + `flush` imediato. Em execuções paralelas, os logs são
  roteados por uma fila compartilhada + listener no processo principal, garantindo que as mensagens apareçam em tempo real no
  Colab.

## Estrutura em módulos Python

O código do notebook foi dividido no pacote `robot_dynamics/`:

- `types.py`: tipos literais para juntas e eixos.
- `models.py`: dataclasses `Joint`, `Link` e `RobotModel`.
- `transforms.py`: transformação DH e utilitário para extrair eixo de rotação/translação.
- `parsing.py`: parsing e validação de eixos e criação de elos a partir de listas de parâmetros.
- `kinematics.py`: cinemática direta e jacobianos espaciais.
- `dynamics.py`: cálculo de energias, Lagrange e matrizes M/C/H/G/τ (com suporte a paralelismo e logs; retorna energia cinética
  total para visualização rápida).
- `__init__.py`: expõe a API principal (`dynamics`, `forward_kinematics`, etc.) e inicializa o `sympy` para impressão.

## Notebook principal

- Use `robot_dynamics_main.ipynb` no Colab (ou localmente) para clonar este repositório automaticamente e executar um exemplo
  mínimo de cinemática/dinâmica para 2 DOF. Atualize a variável `REPO_URL` no topo do notebook para apontar para o repositório
  Git que deseja trabalhar.
