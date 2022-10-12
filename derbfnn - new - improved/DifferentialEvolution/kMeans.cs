using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DifferentialEvolution
{
	class kMeans
	{
		private double[][] centers;
		private int[] partition;
		private int[] dppc;
		private double sse;
		private double[] ssepc;

		private DataSet data;
		private int d;
		private int n;

		private Random rg;

		public kMeans(DataSet dat)
		{
			centers = null;
			dppc = null;
			sse = 0;

			data = dat;
			d = data.Dimensionality;
			n = data.Size;

			partition = new int[n];

			rg = new Random();
		}

		public double Run(int k, int iter, double eps)
		{
			double[][] prevCenters = new double[k][];
			for (int i = 0; i < k; i++)
				prevCenters[i] = new double[d];

			bool conv = true;
			Init(k);
			for (int i = 0; i < iter; i++)
			{
				for (int j = 0; j < k; j++)
					centers[j].CopyTo(prevCenters[j], 0);

				Mdp(k);
				Uc(k);

				conv = true;
				for (int j = 0; j < k; j++)
				{
					for (int l = 0; l < d; l++)
						if (Math.Abs(prevCenters[j][l] - centers[j][l]) > eps)
						{
							conv = false;
							break;
						}
					if (!conv)
						break;
				}

				if (conv)
					break;
			}

			Mdp(k);
			return sse;
		}

		private void Init(int k)
		{
			sse = 0;
			ssepc = new double[k];
			dppc = new int[k];
			centers = new double[k][];
			for (int i = 0; i < k; i++)
				centers[i] = new double[d];

			List<int> irnd = new List<int>();
			for (int i = 0; i < n; i++)
				irnd.Add(i);
			for (int i = 0; i < k; i++)
			{
				int r = rg.Next(0, irnd.Count);
				for (int j = 0; j < d; j++)
					centers[i][j] = data[irnd[r]][j];

				irnd.RemoveAt(r);
			}
		}

		private void Mdp(int k)
		{
			sse = 0;
			for (int i = 0; i < k; i++)
				ssepc[i] = dppc[i] = 0;

			double dmin, tmp;
			int c;
			for (int i = 0; i < n; i++)
			{
				dmin = L2norm(data[i], centers[0]);
				c = 0;
				for (int j = 1; j < k; j++)
				{
					tmp = L2norm(data[i], centers[j]);
					if (tmp < dmin)
					{
						dmin = tmp;
						c = j;
					}
				}

				partition[i] = c;
				dppc[c]++;

				ssepc[c] += dmin * dmin;
			}

			for (int i = 0; i < k; i++)
			{
				if (dppc[i] == 0)
				{
					int ssemax = 0;
					for (int j = 1; j < k; j++)
						if (ssepc[j] > ssepc[ssemax])
							ssemax = j;

					int fdp = 0;
					double dmax = 0;
					double t;
					for (int j = 0; j < n; j++)
						if (partition[j] == ssemax)
						{
							t = L2norm(data[j], centers[ssemax]);
							if (t > dmax)
							{
								fdp = j;
								dmax = t;
							}
						}

					partition[fdp] = i;
					dppc[i]++;
					ssepc[ssemax] -= dmax * dmax;

					for (int j = 0; j < d; j++)
						centers[i][j] = data[fdp][j];
				}
			}

			for (int i = 0; i < k; i++)
				sse += ssepc[i];
		}

		private void Uc(int k)
		{
			for (int i = 0; i < k; i++)
				for (int j = 0; j < d; j++)
					centers[i][j] = 0;

			for (int i = 0; i < n; i++)
				for (int j = 0; j < d; j++)
					centers[partition[i]][j] += data[i][j];

			for (int i = 0; i < k; i++)
				for (int j = 0; j < d; j++)
					centers[i][j] /= dppc[i];
		}

		private double L2norm(double[] v1, double[] v2)
		{
			double result = 0;
			for (int i = 0; i < d; i++)
			{
				result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
			}
			return Math.Sqrt(result);
		}

		public double[][] getCenters()
		{
			return centers;
		}

		public double SSE { get { return sse; } }
	}
}
