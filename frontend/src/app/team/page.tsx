import TopNav from "../components/TopNav";
import TeamCard from "../components/Card";

export default function Team() {
    return (
        <main className="flex h-screen flex-col items-center p-12 m-2">
            <TopNav />
            <div className="grid w-2/3 py-10 grid-cols-2 h-screen justify-center items-center">
                <TeamCard name="Dimitrios Psaltos" title="Research Data Scientist"/>
                <TeamCard name="Marguerite Morgan" title="Data Scientist"/>
                <TeamCard name="Arun Surendranath" title="Project Manager"/>
                <TeamCard name="Milan Dean" title="Full-stack Engineer"/>
            </div>
        </main>
    );
}