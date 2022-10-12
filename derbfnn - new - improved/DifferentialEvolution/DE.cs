using System;
using System.Collections.Generic;
using System.Text;

namespace DifferentialEvolution
{
    enum DESTRATEGY
    {
        RAND,   //DE/RAND/1/BIN -- default
        BEST,   //DE/BEST/1/BIN
		TBEST
    }

    class DE
    {
        private double[][]  currPop;
        private double[][]  newPop;
        private double[]    currSolCosts;
        private double[]    newSolCosts;
        private int[]       currSize;
        private int[]       newSize;

		private int         vsize;
		private int         kappa;
        private int         n;
        private int         d;
		private int			c;
        private double[]    maxBound;
        private double[]    minBound;

        private double      maxCost, minCost, avgCost;
        private int         maxIdx, minIdx;

		private int			iterationBestFound;

		private DataSet     data;	

        private Random      randGen;
		private int			seed;
        private DESTRATEGY  strategy;

        private double lambda;

		private double[] performance;
		private double[] kValues;

        public DE(DataSet datT, DESTRATEGY _destragety)
        {
            randGen         = new Random();
            strategy        = _destragety;

            currPop         = newPop = null;
            currSolCosts    = newSolCosts = null;

			data			= datT;
			n               = data.Size;
			d               = data.Dimensionality;
			c				= data.Classes;
			vsize           = 0;
			kappa           = 0;

            lambda = 0.01 / (2 * c);

			iterationBestFound = 0;
        }

        public double Run(int np, double cr, double f, int iter, int kmin, int kmax)
        {
			this.performance = new double[iter];
			this.kValues = new double[iter];

			//kmin = data.Classes;
			//Initialize(np, kmin, kmax);
			GenerateInitialPopulation2(np, kmin, kmax);
            CalcPopStats();
			double bestCost = currSolCosts[minIdx];
            for (int i = 1; i <= iter; i++)
            {
                for (int j = 0; j < np; j++)
                {
                    GenTrail(j, cr, f);
                }

                for (int j = 0; j < np; j++)
                {
                    newSolCosts[j] = CalcSolCost(newPop[j], kmin, out newSize[j]);
                    if (newSolCosts[j] <= currSolCosts[j])
                    {
                        currSolCosts[j] = newSolCosts[j];
                        currSize[j] = newSize[j];
                        newPop[j].CopyTo(currPop[j], 0);
                    }
                }
                CalcPopStats();
				

                if (i == 1 || currSolCosts[minIdx] < bestCost)
                {
                    bestCost = currSolCosts[minIdx];
                    iterationBestFound = i;
                }

				/* Za review HAIS:
				 */
				this.performance[i - 1] = bestCost;
				this.kValues[i - 1] = currSize[minIdx];


				if (i == (int)(iter * 0.25))
				{
					int tempsize = (int)Math.Ceiling(kmin * 1.2);
					if (tempsize <= currSize[minIdx])
						kmin = tempsize;

					tempsize = (int)Math.Floor(kmax * 0.8);
					if (tempsize >= currSize[minIdx])
						kmax = tempsize;
					ReduceKmax(np, kmax, kmin);
                    //lambda *= 0.95;
					cr *= 0.25;

					//Console.Write("{0} {1}\t", kmin, kmax);
				}
				if (i == (int)(iter * 0.5))
				{
					int tempsize = (int)Math.Ceiling(kmin * 1.2);
					if (tempsize <= currSize[minIdx])
						kmin = tempsize;

					tempsize = (int)Math.Floor(kmax * 0.8);
					if (tempsize >= currSize[minIdx])
						kmax = tempsize;
					ReduceKmax(np, kmax, kmin);
                    //lambda *= 0.95;
					cr *= 0.25;

					//Console.Write("{0} {1}\t", kmin, kmax);
				}
				if (i == (int)(iter * 0.75))
				{
					int tempsize = (int)Math.Ceiling(kmin * 1.2);
					if (tempsize <= currSize[minIdx])
						kmin = tempsize;

					tempsize = (int)Math.Floor(kmax * 0.8);
					if (tempsize >= currSize[minIdx])
						kmax = tempsize;
					ReduceKmax(np, kmax, kmin);
                    //lambda *= 0.95;
					cr *= 0.25;

					//Console.WriteLine("{0} {1}", kmin, kmax);
				}

				//cr -= (0.9 - 0.1) / iter;
                //cr /= Math.Pow(9, 1 / iter);

                //Console.WriteLine("min: {0}\tmax: {1}\tavg: {2}", minCost, maxCost, avgCost);
            }

            return currSolCosts[minIdx];
        }

		private void ReduceKmax(int np, int newKmax, int kmin)
		{
			int tmpK = newKmax;

			int size = tmpK * (d + 2);
			double[][] tmpCurrPop = new double[np][];
			double[][] tmpNewPop = new double[np][];
			for (int i = 0; i < np; i++)
			{
				tmpCurrPop[i] = new double[size];
				tmpNewPop[i] = new double[size];

				double[][] centers = ActiveCenters(currPop[i]);
				double[] widths = ObtainWidths(currPop[i]);
				int l = widths.Length;
				for (int j = 0; j < size; j++)
				{
					if (j < tmpK)
						tmpCurrPop[i][j] = tmpNewPop[i][j] = randGen.NextDouble() * 0.5;
					else
						tmpCurrPop[i][j] = tmpNewPop[i][j] = minBound[j] + randGen.NextDouble() * (maxBound[j] - minBound[j]);

				}

				if (l > tmpK)
					l = tmpK;

				for (int j = 0; j < l; j++)
				{
					tmpCurrPop[i][j] = tmpNewPop[i][j] = 0.5 + randGen.NextDouble() * 0.5;
					tmpCurrPop[i][tmpK * (d + 1) + j] = tmpNewPop[i][tmpK * (d + 1) + j] = widths[j];

					for (int r = 0; r < d; r++)
						tmpCurrPop[i][tmpK + j * d + r] = tmpNewPop[i][tmpK + j * d + r] = centers[j][r];
				}
			}

			currPop = tmpCurrPop;
			newPop = tmpNewPop;
			kappa = tmpK;
			vsize = size;

			for (int i = 0; i < np; i++)
			{
				currSolCosts[i] = newSolCosts[i] = CalcSolCost(currPop[i], kmin, out currSize[i]);
				newSize[i] = currSize[i];
			}
		}

		private void Initialize(int np, int kmin, int kmax)
        {
			//seed = randGen.Next();
            kappa = kmax;
			vsize = kappa * (d + 2);

            maxBound = new double[vsize];
            minBound = new double[vsize];
            for (int i = 0; i < vsize; i++)
            {
                maxBound[i] = 1;
                minBound[i] = 0;
            }


            currPop = new double[np][];
            newPop = new double[np][];
            currSolCosts = new double[np];
            newSolCosts = new double[np];
            currSize = new int[np];
            newSize = new int[np];

            for (int i = 0; i < np; i++)
            {
                currPop[i] = new double[vsize];
                newPop[i] = new double[vsize];
                for (int j = 0; j < vsize; j++)
                {
                    currPop[i][j] = minBound[j] + randGen.NextDouble() * (maxBound[j] - minBound[j]);
                }
                currSolCosts[i] = CalcSolCost(currPop[i], kmin, out currSize[i]);
                newSize[i] = currSize[i];
            }
        }

		private void GenerateInitialPopulation(int np, int kmin, int kmax)
		{
			kappa = kmax;
			vsize = kappa * (d + 2);

			maxBound = new double[vsize];
			minBound = new double[vsize];
			for (int i = 0; i < vsize; i++)
			{
				maxBound[i] = 1;
				minBound[i] = 0;
			}

			currPop = new double[np][];
			newPop = new double[np][];
			currSolCosts = new double[np];
			newSolCosts = new double[np];
            currSize = new int[np];
            newSize = new int[np];

			for (int i = 0; i < np; i++)
			{
				currPop[i] = new double[vsize];
				newPop[i] = new double[vsize];
				for (int j = 0; j < vsize; j++)
				{
					currPop[i][j] = newPop[i][j] =
						minBound[j] + randGen.NextDouble() * (maxBound[j] - minBound[j]);
				}
			}

			kMeans km = new kMeans(data);
			int cnt = 0;
			for (int k = kmin; k <= kmax; k++)
			{
				km.Run(k, 20, 0.01);
				double[][] centers = km.getCenters();
				for (int i = 0; i < k; i++)
					for (int j = 0; j < d; j++)
						currPop[cnt][kmax + i * d + j] = newPop[cnt][kmax + i * d + j] = centers[i][j];
				for (int i = 0; i < kmax; i++)
				{
					if (i < k)
						currPop[cnt][i] = newPop[cnt][i] = 0.5 + randGen.NextDouble() * 0.5;
					else
						currPop[cnt][i] = newPop[cnt][i] = randGen.NextDouble() * 0.5;
				}

				double[] widths = CalcWidths4(centers);
				for (int i = 0; i < k; i++)
					currPop[cnt][kappa * (d + 1) + i] = newPop[cnt][kappa * (d + 1) + i] = widths[i];

				cnt++;
			}

            for (int i = 0; i < np; i++)
            {
                currSolCosts[i] = newSolCosts[i] = CalcSolCost(currPop[i], kmin, out currSize[i]);
                newSize[i] = currSize[i];
            }
		}

        private void GenerateInitialPopulation2(int np, int kmin, int kmax)
        {
            kappa = kmax;
            vsize = kappa * (d + 2);

            maxBound = new double[vsize];
            minBound = new double[vsize];
            for (int i = 0; i < vsize; i++)
            {
                maxBound[i] = 1;
                minBound[i] = 0;
            }

            currPop = new double[np][];
            newPop = new double[np][];
            currSolCosts = new double[np];
            newSolCosts = new double[np];
            currSize = new int[np];
            newSize = new int[np];

            for (int i = 0; i < np; i++)
            {
                currPop[i] = new double[vsize];
                newPop[i] = new double[vsize];
                for (int j = 0; j < vsize; j++)
                {
                    currPop[i][j] = newPop[i][j] =
                        minBound[j] + randGen.NextDouble() * (maxBound[j] - minBound[j]);
                }
            }

            kMeans km = new kMeans(data);
            int cnt = 0;
            int part = (int)(np / 2.0);
            for (int c = 0; c < part; c++)
            {
                int k = randGen.Next(kmin, kmax);
                km.Run(k, 20, 0.01);
                double[][] centers = km.getCenters();
                for (int i = 0; i < k; i++)
                    for (int j = 0; j < d; j++)
                        currPop[cnt][kmax + i * d + j] = newPop[cnt][kmax + i * d + j] = centers[i][j];
                for (int i = 0; i < kmax; i++)
                {
                    if (i < k)
                        currPop[cnt][i] = newPop[cnt][i] = 0.5 + randGen.NextDouble() * 0.5;
                    else
                        currPop[cnt][i] = newPop[cnt][i] = randGen.NextDouble() * 0.5;
                }

                double[] widths = CalcWidths4(centers);
                for (int i = 0; i < k; i++)
                    currPop[cnt][kappa * (d + 1) + i] = newPop[cnt][kappa * (d + 1) + i] = widths[i];

                cnt++;
            }

            for (int i = 0; i < np; i++)
            {
                currSolCosts[i] = newSolCosts[i] = CalcSolCost(currPop[i], kmin, out currSize[i]);
                newSize[i] = currSize[i];
            }
        }

		private double[] CalcWidths(double[][] centers)
		{
			int k = centers.Length;
			double[] widths = new double[k];

			double dmax = 0, tmp;
			for (int i = 0; i < k; i++)
				for (int j = 0; j < k; j++)
					if (i != j)
					{
						tmp = L2norm(centers[i], centers[j]);
						if (i == 0 || dmax < tmp)
							dmax = tmp;
					}

			double w = dmax / Math.Sqrt(2 * k);
			for (int i = 0; i < k; i++)
				widths[i] = w;

			return widths;
		}

        private double[] CalcWidths2(double[][] centers, double b)
        {
            int k = centers.Length;
            double[] widths = new double[k];

            double dmin = 0, tmp;
            bool f = true;
            for (int i = 0; i < k; i++)
            {
                f = true;
                for (int j = 0; j < k; j++)
                    if (i != j)
                    {
                        tmp = L2norm(centers[i], centers[j]);
                        if (f || tmp < dmin)
                            dmin = tmp;
                        f = false;
                    }

                widths[i] = dmin * b;
            }

            return widths;
        }

        private double[] CalcWidths3(double[][] centers)
        {
            int k = centers.Length;
            double[] widths = new double[k];

            double avg = 0;
            for (int i = 0; i < k - 1; i++)
                for (int j = i + 1; j < k; j++)
                    avg += L2norm(centers[i], centers[j]);

            avg /= (k * (k - 1)) / 2.0;
            for (int i = 0; i < k; i++)
                widths[i] = avg;

            return widths;
        }

        private double[] CalcWidths4(double[][] centers)
        {
            int k = centers.Length;
            double[] widths = new double[k];

            double dmin = 0, tmp;
            bool f = true;
            for (int i = 0; i < k; i++)
            {
                f = true;
                for (int j = 0; j < k; j++)
                    if (i != j)
                    {
                        tmp = L2norm(centers[i], centers[j]);
                        if (f || tmp < dmin)
                            dmin = tmp;
                        f = false;
                    }

                widths[i] = dmin * (1 + randGen.NextDouble());
            }

            return widths;
        }

        private double CalcSolCost(double[] v, int kmin, out int size)
		{
            double[][] centers	= ActiveCenters(v, kmin);
            double[] widths		= ObtainWidths(v);

            size = widths.Length;

            RBFNN rbfnn = new RBFNN(centers, widths, data);
            rbfnn.PInvWeights();
			/*
            return rbfnn.sumSquaredError(data) / data.Size / data.Classes + lambda * widths.Length;
			return rbfnn.sumSquaredError(data) / data.Size + lambda * widths.Length;
			return rbfnn.misclassificationRate(data) / data.Size + lambda * widths.Length;
			return rbfnn.sumSquaredError(data) / data.Size / data.Classes + 0.005 * widths.Length;
			 */
            return rbfnn.sumSquaredError(data) / data.Size / data.Classes + lambda * widths.Length;
            //return Sig(rbfnn.sumSquaredError(data) / data.Size / data.Classes) + lambda * widths.Length;


			//double tmp = Math.Log(rbfnn.sumSquaredError(data) / data.Size, 2) * data.Size;
			//return tmp + widths.Length * 4;
		}

		private double[][] ActiveCenters(double[] v, int kmin)
		{
            List<int> tmp = new List<int>();
            for (int i = 0; i < kappa; i++)
                if (v[i] >= 0.5)
                    tmp.Add(i);

            int r;
            while (tmp.Count < kmin)
            {
                r = randGen.Next(0, kappa);
                if (!tmp.Contains(r))
                {
                    tmp.Add(r);
                    v[r] = 0.5 * (1 + randGen.NextDouble());
                }
            }

            int m = tmp.Count;
            double[][] centers = new double[m][];
            for (int i = 0; i < m; i++)
            {
                centers[i] = new double[d];
                for (int j = 0; j < d; j++)
                centers[i][j] = v[kappa + tmp[i] * d + j];
            }

			return centers;
		}

        private double[] ObtainWidths(double[] v)
        {
            List<int> tmp = new List<int>();
            for (int i = 0; i < kappa; i++)
                if (v[i] >= 0.5)
                    tmp.Add(i);

            double[] widths = new double[tmp.Count];
            for (int i = 0; i < tmp.Count; i++)
                widths[i] = v[kappa * (1 + d) + tmp[i]];

            return widths;
        }

        private void GenTrail(int idx, double cr, double f)
        {
            int np = currPop.Length;
            int r0, r1, r2;

            do
            {
                r0 = randGen.Next(0, np);
            } while (r0 == idx);

            do
            {
                r1 = randGen.Next(0, np);
            } while (r1 == r0 || r1 == idx);

            do
            {
                r2 = randGen.Next(0, np);
            } while (r2 == r1 || r2 == r0 || r2 == idx);

            int     randI = randGen.Next(0, vsize);
            double  diff;

            #region STDDE
            for (int i = 0; i < vsize; i++)
            {
                if (randGen.NextDouble() <= cr || i == randI)
                {
                    if (strategy == DESTRATEGY.BEST)
                        diff = currPop[minIdx][i] + f * (currPop[r1][i] - currPop[r2][i]);
                    else if (strategy == DESTRATEGY.TBEST)
                    {
                        int tmp;
                        if (currSolCosts[r1] < currSolCosts[r0])
                        {
                            tmp = r0;
                            r0 = r1;
                            r1 = tmp;
                        }
                        if (currSolCosts[r0] < currSolCosts[r2])
                        {
                            tmp = r0;
                            r0 = r2;
                            r2 = tmp;
                        }
                        diff = currPop[r0][i] + f * (currPop[r1][i] - currPop[r2][i]);
                    }
                    else
                        diff = currPop[r0][i] + f * (currPop[r1][i] - currPop[r2][i]);

                    if (diff < minBound[i])
                        diff = currPop[r0][i] + randGen.NextDouble() * (minBound[i] - currPop[r0][i]);
                    if (diff > maxBound[i])
                        diff = currPop[r0][i] + randGen.NextDouble() * (maxBound[i] - currPop[r0][i]);

                    newPop[idx][i] = diff;
                }
                else
                    newPop[idx][i] = currPop[idx][i];
            }
            #endregion

            #region NEWDE
            //for (int i = 0; i < vsize; i++)
            //{
            //    if (randGen.NextDouble() <= cr || i == randI)
            //    {
            //        if (i >= 20 && i < 20 * (1 + d))
            //        {
            //            for (int j = 0; j < d; j++)
            //                newPop[idx][i + j] = currPop[r0][i + j] + f * (currPop[r1][i + j] - currPop[r2][i + j]);

            //            i += d;
            //        }
            //        else
            //            newPop[idx][i] = currPop[r0][i] + f * (currPop[r1][i] - currPop[r2][i]);

            //        if (newPop[idx][i] < minBound[i])
            //            newPop[idx][i] = currPop[r0][i] + randGen.NextDouble() * (minBound[i] - currPop[r0][i]);
            //        if (newPop[idx][i] > maxBound[i])
            //            newPop[idx][i] = currPop[r0][i] + randGen.NextDouble() * (maxBound[i] - currPop[r0][i]);
            //    }
            //    else
            //    {
            //        if (i >= 20 && i < 20 * (1 + d))
            //        {
            //            for (int j = 0; j < d; j++)
            //                newPop[idx][i + j] = currPop[idx][i + j];

            //            i += d;
            //        }
            //        else
            //            newPop[idx][i] = currPop[idx][i];
            //    }
            //}
            #endregion
        }

        private void CalcPopStats()
        {
            maxCost = minCost = avgCost = currSolCosts[0];
            maxIdx = minIdx = 0;
            double totalCost = 0;

            int i = 0;
            foreach (double c in currSolCosts)
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

        private double L2norm(double[] v1, double[] v2)
        {
            double result = 0;
            for (int i = 0; i < v1.Length; i++)
            {
                result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            }
            return Math.Sqrt(result);
        }

		private double[][] ActiveCenters(double[] v)
		{
			List<int> tmp = new List<int>();
			for (int i = 0; i < kappa; i++)
				if (v[i] >= 0.5)
					tmp.Add(i);

			int m = tmp.Count;
			double[][] centers = new double[m][];
			for (int i = 0; i < m; i++)
			{
				centers[i] = new double[d];
				for (int j = 0; j < d; j++)
					centers[i][j] = v[kappa + tmp[i] * d + j];
			}

			return centers;
		}

		public double[][] getCenters()
		{
			return ActiveCenters(currPop[minIdx]);
		}

		public double[] getWidths()
		{
            return ObtainWidths(currPop[minIdx]);
		}

		public double[] Performance { get { return this.performance; } }

		public double[] KValues { get { return this.kValues; } }

		public int Seed { get { return seed; } }

		public int IterationBestFound { get { return iterationBestFound; } }
        public double MinSolCost { get { return currSolCosts[minIdx]; } }

        private double Sig(double x)
        {
            if (x < -50)
                return 0;
            else if (x > 50)
                return 1;
            return 1 / (1 + Math.Exp(-x));
        }
    }
}
