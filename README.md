# cgm-analytics
Predictive Diets using Continuous Glucose Monitoring 

# Requirements

In order to run the Next.js application locally, you will first need to ensure you have node version >16.0.0 found [here](https://nodejs.org/en), and make sure that's stored in your $PATH.

In order to check and see if node was automatically setup in your environment path, you can run the following command:

```
node --version
```

If this does not return a version of node, then you need to manually add it to your system's path. You can do this on Mac by going into the terminal and typing in the following command:

```
nano ~/.zshrc
export PATH="/usr/local/bin:$PATH"
```

Once the repository has been downloaded, navigate into the cgm-analytics folder containing the _package.json_ file and run the following command:

```
npm install
```

This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/basic-features/font-optimization) to automatically optimize and load Inter, a custom Google Font.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js/) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.
