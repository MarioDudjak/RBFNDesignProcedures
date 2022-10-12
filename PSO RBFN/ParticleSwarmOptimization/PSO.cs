using System;
using System.Collections.Generic;
using System.Text;

namespace ParticleSwarmOptimization
{
    //gbest PSO
    class PSO
    {
        private double[][]  positions;
        private double[][]  velocities;
        private double[]    currSolCosts;
        private double[]    pBestCosts;
        private double[][]  pBests;

		private int			kappa;
		private int			vSize;

        private double      wmin;
        private double      wmax;
		private double[]	maxBound;
		private double[]	minBound;

        private double      maxCost, minCost, avgCost;
        private int         maxIdx, minIdx;

        private Random      randGen;

        private DataSet     data;
		private int			d;

        public PSO(DataSet _data)
        {
            randGen         = new Random();

            data            = _data;
			d				= data.Dimensionality;

            positions       = velocities = pBests = null;
            currSolCosts    = pBestCosts = null;

            ///////////
            wmin = 0.4;
            wmax = 0.9;
            ///////////
        }

        public double Run(int ns, double w, double c1, double c2, int iter, int kmin, int kmax)
        {
			kappa = kmax;
            vSize   = kappa * (d + 2);

            positions   = new double[ns][];
            velocities  = new double[ns][];
            pBests      = new double[ns][];
            for (int i = 0; i < ns; i++)
            {
                positions[i]    = new double[vSize];
                velocities[i]   = new double[vSize];
                pBests[i]       = new double[vSize];
            }

            currSolCosts    = new double[ns];
            pBestCosts      = new double[ns];

			minBound = new double[vSize];
			maxBound = new double[vSize];
			for (int i = 0; i < vSize; i++)
			{
				maxBound[i] = 1;
				minBound[i] = 0;
				if (i % (d + 2) == 0)
				{
					minBound[i] = 0;
					maxBound[i] = 1;
				}
			}

            InitPositions(ns, kmax);
            CalcSwarmStats();

            for (int i = 0; i < iter; i++)
            {
                //S. Paterlini, T. Krink,
                //Differential evolution and particle swarm optimisation in partitional clustering
                //Computational Statistics & Data Analysis 50 (2006) 1220 – 1247
                //////////////////////////////////////////////////////////////////////////////////
                w -= (wmax - wmin) / iter;
                //////////////////////////////////////////////////////////////////////////////////
                
                for (int j = 0; j < ns; j++)
                {
                    UpdateVelPos(j, w, c1, c2);                    
                }

				for (int j = 0; j < ns; j++)
				{
					currSolCosts[j] = CalcSolCost(positions[j], kmin);
					if (currSolCosts[j] < pBestCosts[j])
					{
						pBestCosts[j] = currSolCosts[j];
						for (int l = 0; l < vSize; l++)
							pBests[j][l] = positions[j][l];
					}
				}

                CalcSwarmStats();
				//Console.WriteLine("min: {0}\tmax: {1}\tavg: {2}", minCost, maxCost, avgCost);
            }

            return pBestCosts[minIdx];
        }

		private double[][] ActiveCenters(double[] p, int kmin)
		{
			List<int> tmp = new List<int>();
			for (int i = 0; i < kappa; i++)
				if (p[i * (d + 2)] > 0.5)
					tmp.Add(i);

			int r;
			while (tmp.Count < kmin)
			{
				r = randGen.Next(0, kappa);
				if (!tmp.Contains(r))
				{
					tmp.Add(r);
					p[r * (d + 2)] = 0.5 + randGen.NextDouble() * 0.5;
				}
			}

			int m = tmp.Count;
			double[][] centers = new double[m][];
			for (int i = 0; i < m; i++)
			{
				centers[i] = new double[d];
				for (int j = 0; j < d; j++)
					centers[i][j] = p[1 + tmp[i] * (d + 2) + j];
			}

			return centers;
		}

		private double[] ObtainWidths(double[] p)
		{
			List<int> tmp = new List<int>();
			for (int i = 0; i < kappa; i++)
				if (p[i * (d + 2)] > 0.5)
					tmp.Add(i);

			double[] widths = new double[tmp.Count];
			for (int i = 0; i < tmp.Count; i++)
			{
				widths[i] = p[1 + d + tmp[i] * (d + 1)];
				if (widths[i] == 0)
					widths[i] = p[1 + d + tmp[i] * (d + 1)] = 1E-5;
			}

			return widths;
		}

        private void UpdateVelPos(int idx, double w, double c1, double c2)
        {
            double          v, vmax;
            for (int i = 0; i < vSize; i++)
            {
                v = w * velocities[idx][i] + c1 * randGen.NextDouble() * (pBests[idx][i] - positions[idx][i]) +
                    c2 * randGen.NextDouble() * (pBests[minIdx][i] - positions[idx][i]);

				//vmax = (maxBound - minBound) / N;
				vmax = 1;
                if (v > vmax)
                    v = vmax;
                if (v < -vmax)
                    v = -vmax;

                velocities[idx][i] = v;

                positions[idx][i] += v;
                //My solution
                //if (positions[idx][i] > maxBound)
                //    positions[idx][i] = pBests[idx][i];
                //if (positions[idx][i] < minBound)
                //    positions[idx][i] = pBests[idx][i];

                //S. Paterlini, T. Krink,
                //Differential evolution and particle swarm optimisation in partitional clustering
                //Computational Statistics & Data Analysis 50 (2006) 1220 – 1247
				if (positions[idx][i] > maxBound[i])
				{
					positions[idx][i] -= 2 * (positions[idx][i] - maxBound[i]);
					velocities[idx][i] = -velocities[idx][i];
				}
				if (positions[idx][i] < minBound[i])
				{
					positions[idx][i] += 2 * (minBound[i] - positions[idx][i]);
					velocities[idx][i] = -velocities[idx][i];
				}

				//if (positions[idx][i] > maxBound[i])
				//{
				//	positions[idx][i] = maxBound[i];
				//	velocities[idx][i] = -velocities[idx][i];
				//}
				//if (positions[idx][i] < minBound[i])
				//{
				//	positions[idx][i] = minBound[i];
				//	velocities[idx][i] = -velocities[idx][i];
				//}
            }
        }

        private double CalcSolCost(double[] p, int kmin)
        {
            double[][] centers = ActiveCenters(p, kmin);
            double[] widths = ObtainWidths(p);

            RBFNN rbfnn = new RBFNN(centers, widths, data);
            rbfnn.PInvWeights();

			double tmp = Math.Log(rbfnn.sumSquaredError(data) / data.Size, 2) * data.Size;
			return tmp + widths.Length * 4;
        }

        private void CalcSwarmStats()
        {
            maxCost = minCost = avgCost = pBestCosts[0];
            maxIdx  = minIdx = 0;

            double totalCost = 0;
            int i = 0;
            foreach (double c in pBestCosts)
            {
                totalCost += c;

                if (c < minCost)
                {
                    minCost = c;
                    minIdx = i;
                }

                if (c > maxCost)
                {
                    maxCost = c;
                    maxIdx = i;
                }

                i++;
            }

            avgCost = totalCost / i;
        }

		private void InitPositions(int ns, int kmin)
		{
			for (int i = 0; i < ns; i++)
			{
                for (int j = 0; j < vSize; j++)
                    velocities[i][j] = 0;

				for (int j = 0; j < kappa; j++)
					positions[i][j * (d + 2)] = 1 * randGen.NextDouble() * 1;

				for (int j = 0; j < kappa; j++)
				{
					int r = randGen.Next(0, data.Size);
					for (int k = 0; k < d; k++)
						positions[i][1 + j * (d + 2) + k] = data[r][k];
				}

				double[] cj = new double[d];
				double[] ck = new double[d];
				for (int j = 0; j < kappa; j++)
				{
					double dt = 0;
					for (int z = 0; z < d; z++)
						cj[z] = positions[i][1 + j * (d + 2) + z];
					for (int k = 0; k < kappa; k++)
						if (j != k)
						{
							for (int z = 0; z < d; z++)
								ck[z] = positions[i][1 + k * (d + 2) + z];
							dt += L2norm(cj, ck);
						}

					positions[i][1 + d + j * (d + 1)] = dt / (kappa - 1);
				}

				currSolCosts[i] = pBestCosts[i] = CalcSolCost(positions[i], kmin);
			}

			for (int i = 0; i < ns; i++)
				for (int j = 0; j < vSize; j++)
					pBests[i][j] = positions[i][j];
		}

		private double L2norm(double[] p1, double[] p2)
		{
			double result = 0;
			for (int i = 0; i < p1.Length; i++)
			{
				result += (p1[i] - p2[i]) * (p1[i] - p2[i]);
			}
			return Math.Sqrt(result);
		}

		private double[][] ActiveCenters(double[] p)
		{
			List<int> tmp = new List<int>();
			for (int i = 0; i < kappa; i++)
				if (p[i * (d + 2)] >= 0.5)
					tmp.Add(i);

			int m = tmp.Count;
			double[][] centers = new double[m][];
			for (int i = 0; i < m; i++)
			{
				centers[i] = new double[d];
				for (int j = 0; j < d; j++)
					centers[i][j] = p[1 + tmp[i] * (d + 2) + j];
			}

			return centers;
		}

		public double[][] getCenters()
		{
			return ActiveCenters(positions[minIdx]);
		}

		public double[] getWidths()
		{
			return ObtainWidths(positions[minIdx]);
		}
    }
}
