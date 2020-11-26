class SkillEvolver(LambdaMuEvolver):
    def __init__(self, *args, **kwargs):
        alpha = kwargs.pop('alpha')
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def infer(self):
        for g_hash, (_, score, age) in self.population.items():
            agent_skills = self.genes[g_hash]
            print(np.array(agent_skills))
            print('score: {}, age: {}'.format(score, age))

    def genRandMap(self):
      # agent_skills = [
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
      #     [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.], ]
        agent_skills = [[0 for i in range(3)] for j in range(8)]

        return agent_skills

    def simulate_game(self,
                        game,
                        agent_skills,
                        n_sim_ticks,
                        child_conn,):
        score = calc_diversity_l2(agent_skills, self.alpha)

        if child_conn:
            child_conn.send(score)

        return score

    def mutate(self, gene):
        agent_skills = copy.deepcopy(gene)
        n_agents = len(agent_skills)
        for i in range(random.randint(1, 5)):
            for j in range(i):
                a_i = random.randint(0, n_agents - 1)
                s_i = random.randint(0, len(agent_skills[0]) - 1)
               #if s_i in [0, 5, 6]:
               #    min_xp = 1000
               #else:
                min_xp = 0
                agent_skills[a_i][s_i] = \
                        min(max(min_xp, agent_skills[a_i][s_i] + random.randint(-100, 100)), 20000)

        if n_agents > 1 and random.random() < 0.05:
            # remove agent
            agent_skills.pop(random.randint(0, n_agents - 1))
            n_agents -= 1
        if 8 > n_agents > 0 and random.random() < 0.05:
            # add agent
            agent_skills.append(copy.deepcopy(agent_skills[random.randint(0, n_agents - 1)]))


        return agent_skills

    def make_game(self, agent_skills):
        return None

    def restore(self, **kwargs):
        pass

def some_examples():
   #print('bad situation')
    agent_skills = [
            [0, 0, 0],
            ]
    ent = calc_diversity(agent_skills, 0.5)

   #print('less bad situation')
    agent_skills = [
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)

    agent_skills = [
            [0, 0, 0],
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)

    agent_skills = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 10],
            ]
    ent = calc_diversity(agent_skills, 0.5)


    print('NMMO: pacifism')
    agent_skills = [
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200.,   0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish and one murder')
    agent_skills = [
    [1200.,  0.,  0. ,  0. ,  0. , 1500.,1500. ,  0. ,  0.],
    [1200. ,  0.,1080.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0., 240.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0., 240.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
    [1200. ,  0.,   0.,   0.,   0.,1500.,1500.,   0.,   0.],
   ]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish')
    agent_skills = [
    [1200.,   0., 120. ,  0.,  0., 1500.,1500. ,  0.,   0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,120.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],]
    ent = calc_diversity(agent_skills)

    print('NMMO: a skirmish, smaller population')
    agent_skills = [
    [1200.,   0., 1200. ,  0.,  0.,1500.,1500. ,  0.,   0.],
    [1200. ,  0. ,  0.  , 240., 0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,120.,1500.,1500.  , 0. ,  0.],
    [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],
#   [1200. ,  0. ,  0.  , 0. ,  0.,1500.,1500.  , 0. ,  0.],]
]
    ent = calc_diversity(agent_skills)

if __name__ == '__main__':
#   some_examples()
    parser= argparse.ArgumentParser()
    parser.add_argument('--experiment-name',
                        default='scratch',
                        help='name of the experiment')
    parser.add_argument('--load',
                        default=False,
                        action='store_true',
                        help='whether or not to load a previous experiment')
    parser.add_argument('--n-pop',
                        type=int,
                        default=12,
                        help='population size')
    parser.add_argument('--lam',
                        type=float,
                        default=1 / 3,
                        help='number of reproducers each epoch')
    parser.add_argument(
                        '--mu',
                        type=float,
                        default=1 / 3,
                        help='number of individuals to cull and replace each epoch')
    parser.add_argument('--inference',
                        default=False,
                        action='store_true',
                        help='watch simulations on evolved maps')
    parser.add_argument('--n-epochs',
                        default=10000,
                        type=int,
                        help='how many generations to evolve')
    parser.add_argument('--alpha',
                        default=0.66,
                        type=float,
                        help='balance between skill and agent entropies')
    args = parser.parse_args()
    n_epochs = args.n_epochs
    experiment_name = args.experiment_name
    load = args.load


    save_path = 'evo_experiment/{}'.format(args.experiment_name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    try:
        evolver_path = os.path.join(save_path, 'evolver')
        with open(evolver_path, 'rb') as save_file:
            evolver = pickle.load(save_file)
            print('loading evolver from save file')
        evolver.restore(trash_data=True)
    except FileNotFoundError as e:
        print(e)
        print('no save file to load')

        evolver = SkillEvolver(save_path,
            n_pop=12,
            lam=1 / 3,
            mu=1 / 3,
            n_proc=12,
            n_epochs=n_epochs,
            alpha=args.alpha,
        )

    if args.inference:
        evolver.infer()
    else:
        evolver.evolve(n_epochs=n_epochs)